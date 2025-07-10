# import os, json, argparse, time, glob
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, roc_auc_score, confusion_matrix,
#     classification_report
# )
# from model_cnn import SimpleCNN

# def print_message(msg):
#     print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# def count_docs_per_query(mats_dir, test_jsonl):
#     """
#     For each query in test_jsonl, count how many q{qid}_d*.pt files exist.
#     """
#     qids = set()
#     with open(test_jsonl, "r") as f:
#         for line in f:

#             rec = json.loads(line)

#             if isinstance(rec, list) and len(rec)==2:
#                 qids.add(rec[0])
#             elif isinstance(rec, dict) and "matrix_file" in rec:
#                 qid = int(rec["matrix_file"].split("_d")[0].lstrip("q"))
#                 qids.add(qid)

#     base = os.path.dirname(__file__)
#     mats_root = os.path.join(base, "..", mats_dir)
#     print_message("Document counts per query:")
#     for qid in sorted(qids):
#         pattern = os.path.join(mats_root, f"q{qid}_d*.pt")
#         files = glob.glob(pattern)
#         print_message(f"  Query {qid }: {len(files)} docs")

# class AllPairsTestDataset(Dataset):
#     """
#     For each test query (in test_pairs.jsonl), include ALL q{qid}_d*.pt
#     in mats_dir and label as 1 if doc ∈ positives, else 0.
#     """
#     def __init__(self, test_jsonl, mats_dir):
#         base      = os.path.dirname(__file__)
#         mats_root = os.path.join(base, "..", mats_dir)


#         qrels = {}
#         with open(test_jsonl, "r") as f:
#             for line in f:
#                 qid, did = json.loads(line)
#                 qrels.setdefault(qid, set()).add(did)


#         self.samples = []
#         for qid, pos_set in qrels.items():
#             pattern = os.path.join(mats_root, f"q{qid}_d*.pt")
#             for path in glob.glob(pattern):
#                 fname = os.path.basename(path)
#                 did   = int(fname.split("_d")[1].split(".pt")[0])
#                 label = 1 if did in pos_set else 0
#                 self.samples.append((fname, label))

#         self.mats = mats_root

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         fname, label = self.samples[idx]
#         mat = torch.load(os.path.join(self.mats, fname), map_location="cpu")
#         if mat.ndim == 2:
#             mat = mat.unsqueeze(0)
#         return mat, torch.tensor(label, dtype=torch.float)

# def main():
#     parser = argparse.ArgumentParser(description="Evaluate CNN as classifier")
#     parser.add_argument("--model",      required=True, help="path to .pt model")
#     parser.add_argument("--test_jsonl", required=True,
#                         help="JSONL of positive test pairs [qid, did]")
#     parser.add_argument("--mats_dir",   default="padded_matrices_cnn",
#                         help="folder with q{qid}_d{did}.pt files")
#     parser.add_argument("--batch_size", type=int, default=32)
#     args = parser.parse_args()

#     count_docs_per_query(args.mats_dir, args.test_jsonl)

#     print_message(f"Loading model from {args.model}")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model  = SimpleCNN().to(device)
#     model.load_state_dict(torch.load(args.model, map_location=device))
#     model.eval()

#     ds = AllPairsTestDataset(args.test_jsonl, args.mats_dir)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
#                     pin_memory=device.type=="cuda")

#     y_true, y_score = [], []
#     with torch.no_grad():
#         for mats, labels in dl:
#             mats   = mats.to(device)
#             logits = model(mats)
#             probs  = torch.sigmoid(logits).cpu().tolist()
#             y_score.extend(probs)
#             y_true .extend(labels.tolist())


#     thr = 0.5
#     print_message(f"Using fixed threshold = {thr:.2f}")
#     y_pred = [1 if p >= thr else 0 for p in y_score]

#     print("\n=== Classification Metrics ===")
#     print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_true, y_pred):.4f}")
#     print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
#     print(f"F1-sc ore : {f1_score(y_true, y_pred):.4f}")
#     try:
#         print(f"ROC-AUC  : {roc_auc_score(y_true, y_score):.4f}")
#     except:
#         pass
#     print("\n=== Classification Report ===")
#     print(classification_report(y_true, y_pred, digits=4))

#     cm = confusion_matrix(y_true, y_pred)
#     print("\nConfusion Matrix:")
#     print(cm)

# if __name__ == "__main__":
#     main()

import os
import json
import argparse
import time
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from tqdm import tqdm

# Assuming model_cnn.py is in the same directory or accessible
from model_cnn import SimpleCNN

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- Configuration (MUST MATCH TRAINING SCRIPT) ---
TARGET_QUERY_LEN      = 32
TARGET_DOC_LEN        = 240
PADDING_VALUE         = 0.0
NORMALIZE_MATRICES    = True
# --- End Configuration ---

def pad_or_truncate_tensor(tensor, target_shape, padding_value):
    """Pads or truncates a 2D tensor to a target shape."""
    target_height, target_width = target_shape
    tensor = tensor[:target_height, :target_width]
    current_shape = tensor.shape
    pad_bottom = max(0, target_height - current_shape[0])
    pad_right = max(0, target_width - current_shape[1])
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=padding_value)
    return tensor

class TripletTestDataset(Dataset):
    """
    Loads exactly the pos/neg pairs in the test_triplets JSONL.
    Each line [qid, pos_id, neg_id] → two samples.
    """
    def __init__(self, test_jsonl, mats_dir, normalize=False):
        self.mats_dir   = mats_dir
        self.normalize  = normalize
        self.target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)

        if not os.path.isdir(mats_dir):
            raise FileNotFoundError(f"Matrices dir not found: {mats_dir}")

        self.samples = []
        with open(test_jsonl) as f:
            for line in f:
                qid, pos_id, neg_id = json.loads(line)
                self.samples.append((qid, pos_id, 1))
                self.samples.append((qid, neg_id, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qid, doc_id, label = self.samples[idx]
        fname = f"q{qid}_d{doc_id}.pt"
        path  = os.path.join(self.mats_dir, fname)
        try:
            mat = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            mat = torch.zeros(*self.target_shape)
        if self.normalize:
            m, s = mat.mean(), mat.std()
            if s > 1e-8:
                mat = (mat - m) / s
        mat = pad_or_truncate_tensor(mat, self.target_shape, PADDING_VALUE).unsqueeze(0)
        return mat, torch.tensor(label, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN classifier")
    # --- MODIFIED: Updated defaults and help text ---
    parser.add_argument("--model",      required=True, help="Path to the trained .pt model file (e.g., cnn_classifier_kidA_fiqa.pt)")
    parser.add_argument("--test_jsonl", required=True, help="Path to the test triplets JSONL file (e.g., test_triplets_fiqa_gt5.jsonl)")
    parser.add_argument("--mats_dir",   required=True, help="Path to the directory with UNPADDED matrices (e.g., qd_matrices_fiqa_untuned)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()

    print_message(f"Loading model from {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- MODIFIED: Must instantiate model with same dimensions as training ---
    model = SimpleCNN(query_len=TARGET_QUERY_LEN, doc_len=TARGET_DOC_LEN).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # --- MODIFIED: Pass normalize flag to dataset ---
    ds = TripletTestDataset(args.test_jsonl, args.mats_dir, normalize=NORMALIZE_MATRICES)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type=="cuda"), num_workers=4)

    y_true, y_score = [], []
    with torch.no_grad():
        for mats, labels in tqdm(dl, desc="Evaluating"):
            mats = mats.to(device)
            logits = model(mats)
            probs = torch.sigmoid(logits).cpu().tolist()
            y_score.extend(probs)
            y_true.extend(labels.tolist())

    thr = 0.5
    print_message(f"Using fixed classification threshold = {thr:.2f}")
    y_pred = [1 if p >= thr else 0 for p in y_score]

    print("\n" + "="*10 + " Classification Metrics " + "="*10)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_true, y_score):.4f}")
    except ValueError as e:
        print(f"ROC-AUC  : Could not be computed ({e})")
    
    print("\n" + "="*10 + " Classification Report " + "="*10)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print(cm)

if __name__ == "__main__":
    main()

# python thesis/cf_run_untuned/eval_class.py     --model thesis/cf_run_untuned/cnn_classifier_in_rainbows.pt     --test_jsonl thesis/cf_run_untuned/test_triplets_cf.jsonl     --mats_dir qd_matrices_cf_untuned  