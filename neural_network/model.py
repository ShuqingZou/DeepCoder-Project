import pickle, sys, os
sys.path.append(os.path.abspath(".."))
import tourch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path

# components same as the successor in enumerative-search
COMPONENTS = [
    "ZIPWITH", "*", "MAP", "SQR", "MUL4", "DIV4", "-",
    "MUL3", "DIV3", "MIN", "+", "SCANL", "SHR", "SHL",
    "MAX", "HEAD", "DEC", "SUM", "doNEG", "isNEG",
    "INC", "LAST", "MINIMUM", "isPOS", "SORT", "FILTER",
    "isODD", "REVERSE", "ACCESS", "isEVEN", "COUNT",
    "TAKE", "MAXIMUM", "DROP",
]

# Integer range constraints (-256 to 255)
INT_MIN = -256 
INT_MAX = 255
VOCAB_OFFSET = 2 # 0: PAD, 1: UNK

# special token IDs
PAD_ID = 0
UNK_ID = 1

# vocabulary size
VOCAB_SIZE = (INT_MAX - INT_MIN + 1) + VOCAB_OFFSET

# hyperparameters specific to DeepCoder NN
EMBEDDING_DIM = 20      
HIDDEN_SIZE = 256       
NUM_LAYERS = 3          
MAX_EXAMPLES = 5
EXAMPLE_MAX_LEN = 20

# tokenizer
def encode_integer(n):
    if n < INT_MIN or n > INT_MAX:
        return UNK_ID
    return (n - INT_MIN) + VOCAB_OFFSET

def proccess_io_examples(example, max_length):
    input_ids = []
    inp_val = example.inputs
    out_val = example.output

    if isinstance(inp_val, list):
        for x in inp_val:
            if isinstance(x, list):
                input_ids.extend([encode_integer(i) for i in x])
            else:
                input_ids.append(encode_integer(x))
    else:
        input_ids.append(encode_integer(inp_val))

    if isinstance(out_val, list):
        input_ids.extend([encode_integer(x) for x in out_val])
    else:
        input_ids.append(encode_integer(out_val))
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    padding_len = max_length - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + [PAD_ID] * padding_len
        
    return input_ids

class DeepCoderDataset(nn.Module):
    def __init__(self, dataset_entries, max_examples = 5, max_length = 20):
        self.entries = dataset_entries
        self.max_examples = max_examples
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        examples_matrix = []
        cur_example = entry.examples[:self.max_examples]
        for example in cur_example:
            encode_example = proccess_io_examples(example, self.max_length)
            examples_matrix.append(encode_example)

        # if fewer than MAX_EXAMPLES
        while len(examples_matrix) < self.max_examples:
            examples_matrix.append([PAD_ID] * self.example_max_len)

        labels = [1.0 if entry.attribute.get(comp, False) else 0.0 for comp in COMPONENTS]
        return {
            # tensor shape (5, 20)
            "input_ids": torch.tensor(examples_matrix, dtype = torch.long), 
            # tensor shape (Num_Components)
            "labels": torch.tensor(labels, dtype = torch.float),
        }
    
class DeepCoderEncoder(nn.Module):
    """
    The Encoder is responsible for:
    1. embedding the input integers.
    2. flattening the sequence of embeddings.
    3. running MLP on each example independently.
    """
    def __init__(self, vocab_size, embedding_dimension, hidden_size, input_length):
        super(DeepCoderEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx = PAD_ID)
        self.flat_dimension = input_length * embedding_dimension
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dimension, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        num_examples = x.size(1)
        
        # [Batch, M, L] -> [Batch, M, L, Emb]
        x_emb = self.embed(x)
        
        # [Batch, M, L * Emb]
        x_flat = x_emb.view(batch_size, num_examples, -1)

        # [Batch * M, L * Emb]
        x_merged = x_flat.view(batch_size * num_examples, -1)

        # [Batch * M, Hidden]
        encoded_flat = self.mlp(x_merged)

        # [Batch, M, Hidden]
        encoded_features = encoded_flat.view(batch_size, num_examples, -1)
        
        return encoded_features
    
class DeepCoderDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(DeepCoderDecoder, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)