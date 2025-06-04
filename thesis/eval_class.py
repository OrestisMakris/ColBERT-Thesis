import os, json, argparse, time, glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from model_cnn import SimpleCNN

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def count_docs_per_query(mats_dir, test_jsonl):
    """
    For each query in test_jsonl, count how many q{qid}_d*.pt files exist.
    """
    qids = set()
    with open(test_jsonl, "r") as f:
        for line in f:

            rec = json.loads(line)

            if isinstance(rec, list) and len(rec)==2:
                qids.add(rec[0])
            elif isinstance(rec, dict) and "matrix_file" in rec:
                qid = int(rec["matrix_file"].split("_d")[0].lstrip("q"))
                qids.add(qid)

    base = os.path.dirname(__file__)
    mats_root = os.path.join(base, "..", mats_dir)
    print_message("Document counts per query:")
    for qid in sorted(qids):
        pattern = os.path.join(mats_root, f"q{qid}_d*.pt")
        files = glob.glob(pattern)
        print_message(f"  Query {qid }: {len(files)} docs")

class AllPairsTestDataset(Dataset):
    """
    For each test query (in test_pairs.jsonl), include ALL q{qid}_d*.pt
    in mats_dir and label as 1 if doc ∈ positives, else 0.
    """
    def __init__(self, test_jsonl, mats_dir):
        base      = os.path.dirname(__file__)
        mats_root = os.path.join(base, "..", mats_dir)


        qrels = {}
        with open(test_jsonl, "r") as f:
            for line in f:
                qid, did = json.loads(line)
                qrels.setdefault(qid, set()).add(did)


        self.samples = []
        for qid, pos_set in qrels.items():
            pattern = os.path.join(mats_root, f"q{qid}_d*.pt")
            for path in glob.glob(pattern):
                fname = os.path.basename(path)
                did   = int(fname.split("_d")[1].split(".pt")[0])
                label = 1 if did in pos_set else 0
                self.samples.append((fname, label))

        self.mats = mats_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        mat = torch.load(os.path.join(self.mats, fname), map_location="cpu")
        if mat.ndim == 2:
            mat = mat.unsqueeze(0)
        return mat, torch.tensor(label, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN as classifier")
    parser.add_argument("--model",      required=True, help="path to .pt model")
    parser.add_argument("--test_jsonl", required=True,
                        help="JSONL of positive test pairs [qid, did]")
    parser.add_argument("--mats_dir",   default="padded_matrices_cnn",
                        help="folder with q{qid}_d{did}.pt files")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    count_docs_per_query(args.mats_dir, args.test_jsonl)

    print_message(f"Loading model from {args.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ds = AllPairsTestDataset(args.test_jsonl, args.mats_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    pin_memory=device.type=="cuda")

    y_true, y_score = [], []
    with torch.no_grad():
        for mats, labels in dl:
            mats   = mats.to(device)
            logits = model(mats)
            probs  = torch.sigmoid(logits).cpu().tolist()
            y_score.extend(probs)
            y_true .extend(labels.tolist())


    thr = 0.5
    print_message(f"Using fixed threshold = {thr:.2f}")
    y_pred = [1 if p >= thr else 0 for p in y_score]

    print("\n=== Classification Metrics ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-sc ore : {f1_score(y_true, y_pred):.4f}")
    try:
        print(f"ROC-AUC  : {roc_auc_score(y_true, y_score):.4f}")
    except:
        pass
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()