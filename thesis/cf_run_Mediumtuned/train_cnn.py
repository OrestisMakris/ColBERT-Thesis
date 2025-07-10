import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from collections import deque

from model_cnn import SimpleCNN

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- Configuration ---
MATRICES_DIR          = "/home/st1084516/ColBERT-Thesis/qd_matrices_cf_Mediumtuned"
TRAIN_DATA_FILE       = "train_triplets_cf_16.jsonl"
VALIDATION_DATA_FILE  = "validation_triplets_cf_16.jsonl"
MODEL_SAVE_PATH       = "cnn_classifier_best_usurum.pt" # Path for the single best model
MODEL_SAVE_DIR        = "recent_models_usurum"          # Directory for the last 15 models

TARGET_QUERY_LEN      = 32
TARGET_DOC_LEN        = 220
PADDING_VALUE         = 0.0

LR                    = 1e-4
EPOCHS                = 55
BATCH_SIZE            = 4
NORMALIZE_MATRICES    = False
# --- End Configuration ---

def pad_or_truncate_tensor(tensor, target_shape, padding_value):
    target_height, target_width = target_shape
    tensor = tensor[:target_height, :target_width]
    current_shape = tensor.shape
    pad_bottom = max(0, target_height - current_shape[0])
    pad_right = max(0, target_width - current_shape[1])
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=padding_value)
    return tensor

class RelevanceDataset(Dataset):
    def __init__(self, jsonl_path, mats_dir, is_triplet_file, normalize=False):
        self.mats_dir = mats_dir
        self.normalize = normalize
        self.target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)
        
        if not os.path.isdir(self.mats_dir):
            raise FileNotFoundError(f"Matrices directory not found: {self.mats_dir}")
        
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    qid, pos_id, neg_id = data
                    self.samples.append({"matrix_file": f"q{qid}_d{pos_id}.pt", "label": 1})
                    self.samples.append({"matrix_file": f"q{qid}_d{neg_id}.pt", "label": 0})
                except (json.JSONDecodeError, ValueError):
                    continue
        
        print_message(f"Loaded {len(self.samples)} samples from {jsonl_path}. Norm: {'ON' if self.normalize else 'OFF'}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        matrix_path = os.path.join(self.mats_dir, sample["matrix_file"])
        label = torch.tensor(sample["label"], dtype=torch.float)
        
        try:
            mat = torch.load(matrix_path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            print(f"ERROR: Matrix file not found: {matrix_path}. Skipping.")
            return torch.zeros(1, *self.target_shape), torch.tensor(0.0, dtype=torch.float)

        if self.normalize:
            mean, std = mat.mean(), mat.std()
            if std > 1e-8:
                mat = (mat - mean) / std
        
        mat = pad_or_truncate_tensor(mat, self.target_shape, PADDING_VALUE)
        mat = mat.unsqueeze(0)
        return mat, label

def main():
    print_message("Starting CNN classification training.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Using device: {device}")

    model = SimpleCNN().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    pwd = os.path.dirname(__file__)
    train_data_path = os.path.join(pwd, TRAIN_DATA_FILE)
    val_data_path = os.path.join(pwd, VALIDATION_DATA_FILE)
    
    model_save_dir = os.path.join(pwd, MODEL_SAVE_DIR)
    os.makedirs(model_save_dir, exist_ok=True)

    train_ds = RelevanceDataset(train_data_path, MATRICES_DIR, is_triplet_file=True, normalize=NORMALIZE_MATRICES)
    val_ds = RelevanceDataset(val_data_path, MATRICES_DIR, is_triplet_file=True, normalize=NORMALIZE_MATRICES)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type=="cuda"), num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type=="cuda"), num_workers=4)

    best_val_acc = 0.0
    last_15_models = deque(maxlen=20)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Training]")
        for mats, labels in train_bar:
            mats, labels = mats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_dl)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            val_bar = tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [Validation]")
            for mats, labels in val_bar:
                mats, labels = mats.to(device), labels.to(device)
                logits = model(mats)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.long().tolist())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print_message(
            f"Epoch {epoch} Summary | Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {acc:.4f}, P: {prec:.4f}, R: {rec:.4f}, F1: {f1:.4f}"
        )

        if acc > best_val_acc:
            best_val_acc = acc
            save_path = os.path.join(pwd, MODEL_SAVE_PATH)
            torch.save(model.state_dict(), save_path)
            print_message(f"New best model saved with Acc: {acc:.4f} to {save_path}")

        # --- Save current epoch model and manage the last 15 ---
        epoch_model_path = os.path.join(model_save_dir, f"cnn_epoch_{epoch}.pt")
        if len(last_15_models) == last_15_models.maxlen:
            model_to_remove = last_15_models[0]
            try:
                os.remove(model_to_remove)
            except OSError as e:
                print_message(f"Error removing old model {model_to_remove}: {e}")
        
        torch.save(model.state_dict(), epoch_model_path)
        last_15_models.append(epoch_model_path)

    print_message("Training complete.")
    print_message(f"Final {len(last_15_models)} models saved in: {model_save_dir}")

if __name__ == "__main__":
    main()


#ok for the testing i need a lot more info three paragprs in a nww syn chapter add to discrybe the above two approahes and on new subdchaoter 3 parapgra dyscripin in the cnn eval how we infrence the model in grreat detail for raking