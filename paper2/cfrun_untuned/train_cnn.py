"""
Train the SimpleCNN relevance classifier using HDF5 matrix files.

Key differences from paper/fact_run/train_cnn.py:
  1. resize_tensor() via bilinear interpolation instead of pad/truncate
  2. Matrices loaded from HDF5 (q{qid}/matrices.h5, dataset "d{doc_id}") stored as float16
  3. float16 → float32 conversion happens lazily at load time
"""
import os
import json
import time
import h5py
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MATRICES_DIR         = "./qd_matrices_CF17_untuned"  # relative to this script's dir
TRAIN_DATA_FILE      = "train_triplets_hard.jsonl"       # relative to this script's dir
VALIDATION_DATA_FILE = "validation_triplets_hard.jsonl"

MODEL_SAVE_PATH      = "cnn_classifier_FinalBosss.pt"      # best-by-val-acc checkpoint
MODEL_SAVE_DIR       = "recent_models_FinalBosss"           # rolling last-20 per-epoch saves

TARGET_QUERY_LEN     = 32
TARGET_DOC_LEN       = 128 #finall boss 128

LR                   = 1e-4
EPOCHS               = 40
BATCH_SIZE           = 4
NORMALIZE_MATRICES   = False
# ---------------------------------------------------------------------------


def resize_tensor(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Bilinear resize of a 2-D float tensor to (target_height, target_width).
    
    Replaces the old pad_or_truncate_tensor: instead of discarding real signal
    (truncation) or adding zeros (padding), we scale the whole matrix so that
    every position in the target grid is informed by the original data.
    """
    return F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),          # (1, 1, H, W)
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)                         # (target_H, target_W)


class RelevanceDataset(Dataset):
    """Loads (matrix, label) pairs from HDF5 files produced by matrices_proce/export_qd_matrices.py."""

    def __init__(self, jsonl_path: str, mats_dir: str, normalize: bool = False):
        self.mats_dir    = mats_dir
        self.normalize   = normalize
        self.target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)

        if not os.path.isdir(self.mats_dir):
            raise FileNotFoundError(f"Matrices directory not found: {self.mats_dir}")

        raw_samples = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    qid, pos_id, neg_id = json.loads(line)
                    raw_samples.append({"qid": qid, "doc_id": pos_id, "label": 1})
                    raw_samples.append({"qid": qid, "doc_id": neg_id, "label": 0})
                except (json.JSONDecodeError, ValueError):
                    continue

        # Filter out samples whose h5 file doesn't exist — avoids zero-fallback noise
        self.samples = []
        skipped = 0
        for s in raw_samples:
            h5_path = os.path.join(self.mats_dir, f"q{s['qid']}", "matrices.h5")
            if os.path.isfile(h5_path):
                self.samples.append(s)
            else:
                skipped += 1

        print_message(
            f"Loaded {len(self.samples)} samples from {jsonl_path} "
            f"({skipped} skipped — missing h5). "
            f"Norm: {'ON' if self.normalize else 'OFF'}."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        qid    = sample["qid"]
        doc_id = sample["doc_id"]
        label  = torch.tensor(sample["label"], dtype=torch.float)

        h5_path = os.path.join(self.mats_dir, f"q{qid}", "matrices.h5")

        try:
            with h5py.File(h5_path, "r") as h5f:
                key = f"d{doc_id}"
                if key not in h5f:
                    raise KeyError(key)
                # float16 numpy → float32 tensor
                mat = torch.from_numpy(h5f[key][:]).float()
        except (FileNotFoundError, KeyError, OSError):
            print_message(f"WARNING: missing {h5_path}::d{doc_id} — using zeros.")
            return torch.zeros(1, *self.target_shape), torch.tensor(0.0, dtype=torch.float)

        if self.normalize:
            mean, std = mat.mean(), mat.std()
            if std > 1e-8:
                mat = (mat - mean) / std

        mat = resize_tensor(mat, self.target_shape)
        mat = mat.unsqueeze(0)   # (1, H, W)
        return mat, label


def main():
    print_message("Starting CNN classification training (paper2 — HDF5 + resize).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Using device: {device}")

    model     = SimpleCNN().to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    pwd            = os.path.dirname(os.path.abspath(__file__))
    train_path     = os.path.join(pwd, TRAIN_DATA_FILE)
    val_path       = os.path.join(pwd, VALIDATION_DATA_FILE)
    model_save_dir = os.path.join(pwd, MODEL_SAVE_DIR)
    os.makedirs(model_save_dir, exist_ok=True)

    train_ds = RelevanceDataset(train_path, MATRICES_DIR, normalize=NORMALIZE_MATRICES)
    val_ds   = RelevanceDataset(val_path,   MATRICES_DIR, normalize=NORMALIZE_MATRICES)

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=4,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=4,
    )

    best_val_acc     = 0.0
    last_20_checkpts = deque(maxlen=20)

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        bar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for mats, labels in bar:
            mats, labels = mats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mats)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_dl)

        # ---- Validation ----
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for mats, labels in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
                mats, labels = mats.to(device), labels.to(device)
                logits = model(mats)
                preds  = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.long().tolist())

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        print_message(
            f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {acc:.4f}  P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f}"
        )

        # ---- Save best by val accuracy ----
        if acc > best_val_acc:
            best_val_acc = acc
            save_path    = os.path.join(pwd, MODEL_SAVE_PATH)
            torch.save(model.state_dict(), save_path)
            print_message(f"  ↑ New best ({acc:.4f}) saved → {save_path}")

        # ---- Rolling last-20 per-epoch saves ----
        epoch_ckpt = os.path.join(model_save_dir, f"cnn_epoch_{epoch}.pt")
        if len(last_20_checkpts) == last_20_checkpts.maxlen:
            old = last_20_checkpts[0]
            try:
                os.remove(old)
            except OSError as e:
                print_message(f"    WARNING: could not remove {old}: {e}")
        torch.save(model.state_dict(), epoch_ckpt)
        last_20_checkpts.append(epoch_ckpt)

    print_message("Training complete.")
    print_message(f"Best val accuracy: {best_val_acc:.4f}")
    print_message(f"Last {len(last_20_checkpts)} checkpoints in: {model_save_dir}")


if __name__ == "__main__":
    main()
