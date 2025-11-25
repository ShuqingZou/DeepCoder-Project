#!/usr/bin/env python3
import pickle, sys, os
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertConfig, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
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

INT_MIN = -512
INT_MAX = 512
VOCAB_OFFSET = 4

# special token ID
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3

VOCAB_SIZE = (INT_MAX - INT_MIN + 1) + VOCAB_OFFSET

# tokenizer
def encode_integer(n):
    if n < INT_MIN or n > INT_MAX:
        return UNK_ID
    return (n - INT_MIN) + VOCAB_OFFSET

def process_entry(entry, max_len = 128):
    input_ids = [CLS_ID]
    for input, output in entry.examples:
        if isinstance(input, list):
            input_ids.extend([encode_integer(x) for x in input])
        else:
            input_ids.append(encode_integer(input))
        input_ids.append(SEP_ID)

        if isinstance(output, list):
            input_ids.extend([encode_integer(x) for x in output])
        else:
            input_ids.append(encode_integer(output))
        input_ids.append(SEP_ID)

        if len(input_ids) >= max_len:
            input_ids = input_ids[:max_len]
            break
    return input_ids

# Dataset class
class DeepCoderIntegerDataset(Dataset):
    def __init__(self, dataset_entries, max_length = 128):
        self.entries = dataset_entries
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        input_ids = process_entry(entry, self.max_length)
        
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [PAD_ID] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        labels = [1.0 if entry.attributes[comp] else 0.0 for comp in COMPONENTS]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }
    
# trainning process
def compute_metrics(prediction):
    logits, labels = prediction
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    f1 = f1_score(labels, predictions, average='micro')
    return {"f1_micro": f1, "accuracy": accuracy_score(labels, predictions)}

if __name__ == "__main__":
    model_dir = Path("models/deepcoder_custom_transformer")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    with open("dataset.pickle", "rb") as f:
        d = pickle.load(f)
    all_data = d.dataset
    # seperate train and test set
    train_entries, val_entries = train_test_split(all_data, test_size=0.1, random_state=42)
    
    train_dataset = DeepCoderIntegerDataset(train_entries, max_length=128)
    val_dataset = DeepCoderIntegerDataset(val_entries, max_length=128)

    print("Initializing Custom Model from Scratch...")
    
    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        num_labels=len(COMPONENTS),
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.1, 
    )

    model = BertForSequenceClassification(config)

    print(f"Model Parameters: {model.num_parameters() / 1e6:.2f} Million")

    args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    trainer.save_model(model_dir)
    print(f"Model saved to {model_dir}")