#!/usr/bin/env python3
import pickle
import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
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

# transfer i/o into prompt
def build_io_text(entry):
    text = []
    for i, (inp, out) in enumerate(entry.examples):
        text.append(f"Example {i}: input={inp}, output={out}")
    return "\n".join(text)


# Dataset class
class DeepCoderTransformerDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        text = build_io_text(entry)

        # label: 34维 multi-label
        labels = [1 if entry.attributes[comp] else 0 for comp in COMPONENTS]

        return {
            "text": text,
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokens["labels"] = labels
    return tokens


if __name__ == "__main__":
    print("Loading dataset...")
    with open("dataset.pickle", "rb") as f:
        d = pickle.load(f)
    data = d.dataset

    train_dataset = DeepCoderTransformerDataset(data)

    model_dir = Path("models/component-model")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = "distilbert-base-uncased"

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(COMPONENTS),
        problem_type="multi_label_classification",
    )

    print("Starting training...")
    args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Model saved to {model_dir}")
