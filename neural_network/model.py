import pickle, sys, os
import time
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

# Integer range constraints
INT_MIN = -256 
INT_MAX = 255
VOCAB_OFFSET = 2

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

    if isinstance(inp_val, tuple):
        for x in inp_val:
            if isinstance(x, tuple):
                input_ids.extend([encode_integer(i) for i in x])
            else:
                input_ids.append(encode_integer(x))
    else:
        input_ids.append(encode_integer(inp_val))

    if isinstance(out_val, tuple):
        input_ids.extend([encode_integer(x) for x in out_val])
    else:
        input_ids.append(encode_integer(out_val))
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    padding_len = max_length - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + [PAD_ID] * padding_len      
    return input_ids

def calculate_score(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    f1 = f1_score(labels_np, preds_np, average='micro')
    acc = accuracy_score(labels_np, preds_np)
    return f1, acc

class DeepCoderDataset(Dataset):
    def __init__(self, dataset, max_examples = 5, max_length = 20):
        self.dataset = dataset
        self.max_examples = max_examples
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        processed_examples = []
        cur_examples = dataset.examples[:self.max_examples]
        for example in cur_examples:
            processed_example = process_io_examples(example, self.max_length)
            processed_examples.append(processed_example)
        while len(processed_examples) < self.max_examples:
            processed_examples.append([PAD_ID] * self.max_length)
        labels = [1.0 if dataset.attribute.get(comp, False) else 0.0 for comp in COMPONENTS]
        return {
            # tensor shape (5, 20)
            "examples": torch.tensor(processed_examples, dtype = torch.long), 
            # tensor shape (omponents true/false)
            "labels": torch.tensor(labels, dtype = torch.float),
        }
    
class DeepCoderEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, input_length):
        super(DeepCoderEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx = PAD_ID)
        self.flat_dim = input_length * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0) #64
        num_examples = x.size(1) #5
        # [batch, 5, 20] -> [batch, 5, 20, 20]
        x = self.embedding(x)
        # [batch, 5, 20, 20] -> [batch, 5, 400]
        x = x.view(batch_size, num_examples, -1)
        # [64, 5, 400] -> [320, 400]
        x = x.view(batch_size * num_examples, -1)
        # [320, 400] -> [320, 256]
        x = self.mlp(x)
        # [320, 256] -> [64, 5, 256]
        x = x.view(batch_size, num_examples, -1)
        return x

class DeepCoderDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(DeepCoderDecoder, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, encoder_features):
        pooled_features, _ = torch.max(encoder_features, dim=1)
        logits = self.classifier(pooled_features)
        
        return logits
    
class DeepCoderModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, input_length, num_classes):
        super(DeepCoderModel, self).__init__()
        self.encoder = DeepCoderEncoder(num_embeddings, embedding_dim, hidden_size, input_length)
        self.decoder = DeepCoderDecoder(hidden_size, num_classes)
    def forward(self, x):
        encoded_features = self.encoder(x)
        logits = self.decoder(encoded_features)
        return logits

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    current_script_path = Path(__file__).resolve()
    project_root = current_script_path.parent.parent

    dataset_path = project_root / "bickle100k.pickle"
    model_path = current_script_path / "models"
    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        d = pickle.load(f)
    if hasattr(d, "dataset"):
        all_data = list(d.dataset)
    else:
        all_data = d 

    print(f"Total dataset loaded")

    # seperate train and test set
    train_dataset, test_dataset = train_test_split(all_data, test_size=0.1, random_state=42)

    processed_train_data = DeepCoderDataset(train_dataset, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)
    processed_test_data = DeepCoderDataset(test_dataset, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)
    
    batch_size = 64
    train_loader = DataLoader(processed_train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(processed_test_data, batch_size=batch_size, shuffle=False)

    model = DeepCoderModel(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        input_length=EXAMPLE_MAX_LEN,
        num_classes=len(COMPONENTS)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train
    num_epochs = 50
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            inputs = batch["examples"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        #test
        model.eval()
        total_test_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["examples"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                total_test_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        
        avg_test_loss = total_test_loss / len(test_loader)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        val_f1, val_acc = calculate_score(all_logits, all_labels)
        
        elapsed = time.time() - start_time

        print(f"{epoch+1} | {avg_train_loss:.4f}     | {avg_test_loss:.4f}     | {val_f1:.4f}   | {val_acc:.4f}   | {elapsed:.0f}s")

        save_name = model_path / f"epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)

    print("models saved")