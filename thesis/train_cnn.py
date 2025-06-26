# import os, json
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# from torch.utils.data import Dataset, DataLoader

# try:
#     from plot_heatmap import print_message
# except ImportError:
#     import time
#     def print_message(msg):
#         print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# from model_cnn import SimpleCNN

# # --- Config ---
# TRAIN_DATA_FILE   = "train_data_balanced.jsonl"
# PADDED_MATRICES   = "padded_matrices_cnn"
# MODEL_SAVE_PATH  = "cnn_classifierrr.pt"
# LR               = 1e-3
# EPOCHS           = 50
# BATCH_SIZE       = 4 #4
# # --------------

# class ClassificationDataset(Dataset):
#     def __init__(self, jsonl_path, mats_dir):
#         base = os.path.dirname(__file__)
#         self.dir = os.path.join(base, "..", mats_dir)
#         if not os.path.isdir(self.dir):
#             raise FileNotFoundError(self.dir)
#         self.samples = []
#         with open(jsonl_path) as f:
#             for ln, line in enumerate(f):
#                 obj = json.loads(line)
#                 if "matrix_file" in obj and obj["label"] in (0,1):
#                     self.samples.append(obj)
#                 else:
#                     print_message(f"Skipping malformed line {ln+1}")
#         print_message(f"Loaded {len(self.samples)} samples")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, i):
#         rec  = self.samples[i]
#         path = os.path.join(self.dir, rec["matrix_file"])

#         mat = torch.load(path, map_location="cpu")
#         # If the saved tensor already has a leading batch dim, drop it:
#         if mat.ndim == 4 and mat.shape[0] == 1:
#             mat = mat.squeeze(0)          # now [C,H,W] or [H,W]

#         # Ensure we have a channel dim:
#         if mat.ndim == 2:                 # [H,W]
#             mat = mat.unsqueeze(0)        # [1,H,W]

#         # Now mat is exactly [C,H,W]. DataLoader will make it [B,C,H,W].
#         label = torch.tensor(rec["label"], dtype=torch.float)
#         return mat, label

# def main():
#     print_message("Starting classification training")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model     = SimpleCNN().to(device)
#     pos_weight_tensor = torch.tensor([1.0], device=device)
#     optimizer = Adam(model.parameters(), lr=LR)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # Adjust pos_weight based on class imbalance
#     #criterion = nn.MSELoss(pos_weight=torch.tensor([0.5], device=device)) 
    
#     pwd        = os.path.dirname(__file__)
#     data_path  = os.path.join(pwd, TRAIN_DATA_FILE)
#     ds         = ClassificationDataset(data_path, PADDED_MATRICES)
#     dl         = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,pin_memory=device.type=="cuda")

#     model.train()
#     for epoch in range(1, EPOCHS+1):
#         total_loss = 0.0
#         from tqdm import tqdm
#         bar = tqdm(enumerate(dl), total=len(dl), desc=f"Epoch {epoch}/{EPOCHS}")
#         for i, (mats, labs) in bar:
#             labs = labs.to(device)
#             mats = mats.to(device)
#             optimizer.zero_grad()
#             logits = model(mats)                # → [B] or [B,1]
#             labs   = labs.view_as(logits)       # match shape
#             loss   = criterion(logits, labs)
#             loss.backward(); optimizer.step()
#             total_loss += loss.item()
#             if (i+1)%10==0:
#                 bar.set_postfix(avg_loss=total_loss/(i+1))
#         avg = total_loss/len(dl)
#         print_message(f"Epoch {epoch} done, avg loss {avg:.4f}")

#     save_path = os.path.join(pwd, MODEL_SAVE_PATH)
#     torch.save(model.state_dict(), save_path)
#     print_message(f"Model saved to {save_path}")

# if __name__=="__main__":
#     main()

import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

try:
    from plot_heatmap import print_message
except ImportError:
    def print_message(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

from model_cnn import SimpleCNN

# --- Config ---
TRAIN_DATA_FILE       = "train_data_balanced.jsonl"
PADDED_MATRICES       = "padded_matrices_cnn"
MODEL_SAVE_PATH       = "cnn_classifier.pt"
LR                    = 1e-3
EPOCHS                = 5
BATCH_SIZE            = 4
VALIDATION_SPLIT      = 0.10  # 10% hold-out
RANDOM_STATE_SPLIT    = 42
POS_CLASS_WEIGHT      = 1.0   # adjust if needed
# --------------

class ClassificationDataset(Dataset):
    def __init__(self, jsonl_path, mats_dir, samples_list=None):
        base = os.path.dirname(__file__)
        self.dir = os.path.join(base, "..", mats_dir)
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(self.dir)

        if samples_list is not None:
            self.samples = samples_list
        else:
            self.samples = []
            with open(jsonl_path) as f:
                for ln, line in enumerate(f):
                    obj = json.loads(line)
                    if "matrix_file" in obj and obj.get("label") in (0,1):
                        self.samples.append(obj)
                    else:
                        print_message(f"Skipping malformed line {ln+1}")
            print_message(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rec = self.samples[i]
        path = os.path.join(self.dir, rec["matrix_file"])
        mat  = torch.load(path, map_location="cpu")

        if mat.ndim == 4 and mat.shape[0] == 1:
            mat = mat.squeeze(0)
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)

        label = torch.tensor(rec["label"], dtype=torch.float)
        return mat, label

def main():
    print_message("Starting classification training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    pos_weight = torch.tensor([POS_CLASS_WEIGHT], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=LR)

    pwd       = os.path.dirname(__file__)
    data_path = os.path.join(pwd, TRAIN_DATA_FILE)

    # Load all samples
    all_samps = []
    with open(data_path) as f:
        for ln, line in enumerate(f):
            obj = json.loads(line)
            if "matrix_file" in obj and obj.get("label") in (0,1):
                all_samps.append(obj)

    print_message(f"Total samples: {len(all_samps)}")

    # Split train/validation
    train_samps, val_samps = train_test_split(
        all_samps,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE_SPLIT,
        stratify=[o["label"] for o in all_samps]
    )
    print_message(f"Train: {len(train_samps)}, Val: {len(val_samps)} samples")

    train_ds = ClassificationDataset(None, PADDED_MATRICES, samples_list=train_samps)
    val_ds   = ClassificationDataset(None, PADDED_MATRICES, samples_list=val_samps)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type=="cuda")
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type=="cuda")
    )

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        from tqdm import tqdm
        train_bar = tqdm(enumerate(train_dl), total=len(train_dl),
                         desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for i, (mats, labs) in train_bar:
            mats = mats.to(device)
            labs = labs.to(device).view(-1)
            optimizer.zero_grad()
            logits = model(mats)
            loss   = criterion(logits, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i+1) % 10 == 0:
                train_bar.set_postfix(avg_loss=total_loss/(i+1))

        avg_train_loss = total_loss / len(train_dl)
        print_message(f"Epoch {epoch} Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mats_v, labs_v in val_dl:
                mats_v = mats_v.to(device)
                logits_v = model(mats_v)
                preds_v  = (logits_v > 0).long().cpu().tolist()
                all_preds.extend(preds_v)
                all_labels.extend(labs_v.long().tolist())

        prec0, prec1 = precision_score(
            all_labels, all_preds,
            labels=[0,1], average=None, zero_division=0
        )
        print_message(
            f"Epoch {epoch} Val precision → "
            f"class 0: {prec0:.4f}, class 1: {prec1:.4f}"
        )

    # Save model
    save_path = os.path.join(pwd, MODEL_SAVE_PATH)
    torch.save(model.state_dict(), save_path)
    print_message(f"Model saved to {save_path}")

if __name__=="__main__":
    main()


