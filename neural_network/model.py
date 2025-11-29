import pickle, sys, os
sys.path.append(os.path.abspath(".."))
import torch
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
    "MUL3", "DIV3", "MIN", "+", "SCANL1", "SHR", "SHL",
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

def process_io_examples(example, max_length):
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

class DeepCoderDataset(Dataset):
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
            encode_example = process_io_examples(example, self.max_length)
            examples_matrix.append(encode_example)

        # if fewer than MAX_EXAMPLES
        while len(examples_matrix) < self.max_examples:
            examples_matrix.append([PAD_ID] * self.max_length)

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
        self.embed = nn.Embedding(vocab_size, embedding_dimension, padding_idx=PAD_ID)
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

    def forward(self, encoder_features):
        """
        input: [batch_size, num_examples, hidden_size]
        output: [batch_size, num_classes]
        """
        pooled_features, _ = torch.max(encoder_features, dim=1)

        logits = self.classifier(pooled_features)
        return logits
    
class DeepCoderModel(nn.Module):
    """
    The main model container that connects Encoder and Decoder.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, input_len):
        super(DeepCoderModel, self).__init__()
        
        self.encoder = DeepCoderEncoder(vocab_size, embedding_dim, hidden_size, input_len)
        self.decoder = DeepCoderDecoder(hidden_size, num_classes)
        
    def forward(self, x):
        # encode the example features
        encoded_features = self.encoder(x)
        # pool and predict
        logits = self.decoder(encoded_features)
        return logits
    
def calculate_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    preds = (probs_np > 0.5).astype(int)
    f1 = f1_score(labels_np, preds, average='micro')
    acc = accuracy_score(labels_np, preds) 
    return f1, acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent
    
    # dataset path: DeepCoder-Project/bickle100k.pickle
    dataset_path = project_root / "bickle100k.pickle"
    model_dir = current_script_path.parent / "models" / "deepcoder_modular_nn"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        d = pickle.load(f)
    all_data = d.dataset

    print(f"Total entries loaded")
    # seperate train and test set
    train_entries, val_entries = train_test_split(all_data, test_size=0.1, random_state=42)

    train_dataset = DeepCoderDataset(train_entries, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)
    val_dataset = DeepCoderDataset(val_entries, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing Neural Network...")

    model = DeepCoderModel(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_size=HIDDEN_SIZE, 
        num_classes=len(COMPONENTS),
        input_len=EXAMPLE_MAX_LEN
    ).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} Million")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 60
    best_f1 = 0.0

    print("Starting training...")
    
    for epoch in range(epochs):
        # train
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # validation
        model.eval()
        val_loss = 0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(input_ids)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        val_f1, val_acc = calculate_metrics(all_logits, all_labels)
        
        print(f"Epoch: {epoch+1:02d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_dir / "nn_model.pth")
            print(f"new best model saved")

    torch.save(model.state_dict(), model_dir / "nn_model.pth")