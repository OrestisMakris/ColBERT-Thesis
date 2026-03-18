"""
Sweep all checkpoint .pt files in a directory and find the best models.

For each checkpoint:
  - Evaluates MAP and MRR at topk = 5,10,20,50,100,250,500,1000
  - Same resize + HDF5 logic as eval_cnn_multi_topk.py

Reports:
  1. Best MAP model  (at each top-k)
  2. Best MRR model  (at each top-k)
  3. Best of both worlds — highest MAP+MRR combined score (at each top-k)

Usage:
  python paper2/cfrun/eval_sweep_best_model.py \
      --models_dir paper2/cfrun/recent_models_triliza \
      --qrels_path paper2/cfrun/test_triplets_hard.jsonl \
      --mats_dir qd_matrices_CF17_untuned/ \
      --candidates_dir ivf_candidates_cf17_untuned/ \
      --topk 5,10,20,50,100,250,500,1000 \
      --batch_size 500
"""
import os
import json
import glob
import h5py
import torch
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F

from model_cnn import SimpleCNN


# ---------------------------------------------------------------------------
# Configuration — must match training
# ---------------------------------------------------------------------------
TARGET_QUERY_LEN   = 32
TARGET_DOC_LEN     = 128
NORMALIZE_MATRICES = False
# ---------------------------------------------------------------------------


def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def resize_tensor(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    return F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Qrels
# ---------------------------------------------------------------------------

def load_qrels(path: str) -> dict:
    qrels = defaultdict(set)
    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if len(data) >= 2:
                    qrels[int(data[0])].add(int(data[1]))
            except Exception:
                continue
    return qrels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_query(qid, model, device, mats_dir, batch_size, candidates_dir):
    target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)
    h5_path = os.path.join(mats_dir, f"q{qid}", "matrices.h5")
    if not os.path.isfile(h5_path):
        return []

    with h5py.File(h5_path, "r") as h5f:
        all_keys = set(h5f.keys())

    if candidates_dir:
        cand_file = os.path.join(candidates_dir, f"q{qid}", f"q{qid}_ivf_candidates.txt")
        if not os.path.exists(cand_file):
            return []
        doc_keys = []
        with open(cand_file) as f:
            for line in f:
                key = f"d{line.strip()}"
                if key in all_keys:
                    doc_keys.append(key)
    else:
        doc_keys = sorted(all_keys, key=lambda k: int(k[1:]))

    if not doc_keys:
        return []

    all_scores = []
    with h5py.File(h5_path, "r") as h5f:
        for i in range(0, len(doc_keys), batch_size):
            batch_keys = doc_keys[i:i + batch_size]
            mats, dids = [], []
            for key in batch_keys:
                try:
                    mat = torch.from_numpy(h5f[key][:]).float()
                    mat = resize_tensor(mat, target_shape)
                    mats.append(mat)
                    dids.append(int(key[1:]))
                except Exception:
                    continue
            if not mats:
                continue
            tensor = torch.stack([m.unsqueeze(0) for m in mats]).to(device)
            with torch.no_grad():
                scores = model(tensor).squeeze().cpu()
            if scores.dim() == 0:
                scores = [scores.item()]
            else:
                scores = scores.tolist()
            all_scores.extend(zip(dids, scores))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in all_scores]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def average_precision(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def reciprocal_rank(retrieved, relevant, k):
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved, relevant, k):
    return sum(1 for d in retrieved[:k] if d in relevant) / k if k > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluate one checkpoint
# ---------------------------------------------------------------------------

def evaluate_checkpoint(model_path, eval_qids, qrels, device, mats_dir,
                         batch_size, candidates_dir, topk_values):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    rankings = {}
    for qid in eval_qids:
        rankings[qid] = score_query(qid, model, device, mats_dir, batch_size, candidates_dir)

    results = {}
    for k in topk_values:
        map_s = mrr_s = p1_s = p5_s = p10_s = 0.0
        for qid in eval_qids:
            ret = rankings[qid]
            rel = qrels.get(qid, set())
            map_s  += average_precision(ret, rel, k)
            mrr_s  += reciprocal_rank(ret, rel, k)
            p1_s   += precision_at_k(ret, rel, 1)
            p5_s   += precision_at_k(ret, rel, 5)
            p10_s  += precision_at_k(ret, rel, 10)
        n = len(eval_qids)
        results[k] = {
            "MAP":  map_s / n,
            "MRR":  mrr_s / n,
            "P@1":  p1_s  / n,
            "P@5":  p5_s  / n,
            "P@10": p10_s / n,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep checkpoints — find best MAP, MRR, and combined model"
    )
    parser.add_argument("--models_dir",    required=True, help="Dir with cnn_epoch_*.pt files")
    parser.add_argument("--qrels_path",    required=True)
    parser.add_argument("--mats_dir",      required=True)
    parser.add_argument("--candidates_dir", default=None)
    parser.add_argument("--topk",          default="5,10,20,50,100,250,500,1000")
    parser.add_argument("--batch_size",    type=int, default=128)
    args = parser.parse_args()

    topk_values = sorted(set(int(k) for k in args.topk.split(",")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Device: {device}")

    qrels = load_qrels(args.qrels_path)

    # Discover checkpoints sorted by epoch number
    ckpts = sorted(
        glob.glob(os.path.join(args.models_dir, "cnn_epoch_*.pt")),
        key=lambda p: int(os.path.basename(p).replace("cnn_epoch_", "").replace(".pt", ""))
    )
    if not ckpts:
        raise RuntimeError(f"No cnn_epoch_*.pt files found in {args.models_dir}")
    print_message(f"Found {len(ckpts)} checkpoints: epoch {os.path.basename(ckpts[0])} → {os.path.basename(ckpts[-1])}")

    # Discover available queries
    available_qids = set()
    for name in os.listdir(args.mats_dir):
        if name.startswith("q") and os.path.isdir(os.path.join(args.mats_dir, name)):
            try:
                available_qids.add(int(name[1:]))
            except ValueError:
                continue
    eval_qids = sorted(set(qrels.keys()) & available_qids)
    print_message(f"Evaluating on {len(eval_qids)} queries.")

    # ---- Sweep ----
    all_results = {}   # {ckpt_path: {k: {metric: value}}}
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        print_message(f"Evaluating {name} …")
        res = evaluate_checkpoint(
            ckpt, eval_qids, qrels, device,
            args.mats_dir, args.batch_size, args.candidates_dir, topk_values
        )
        all_results[ckpt] = res

    # ---- Find bests per top-k ----
    print_message("\nSweep complete. Summary:")

    col_w = 14
    ckpt_labels = {c: os.path.basename(c).replace("cnn_epoch_", "ep").replace(".pt", "") for c in ckpts}

    for k in topk_values:
        best_map_ckpt  = max(all_results, key=lambda c: all_results[c][k]["MAP"])
        best_mrr_ckpt  = max(all_results, key=lambda c: all_results[c][k]["MRR"])
        best_both_ckpt = max(all_results, key=lambda c: all_results[c][k]["MAP"] + all_results[c][k]["MRR"])

        bm  = all_results[best_map_ckpt][k]
        bmr = all_results[best_mrr_ckpt][k]
        bb  = all_results[best_both_ckpt][k]

        print(f"\n{'='*70}")
        print(f"  TOP-K = {k}")
        print(f"{'='*70}")
        header = f"  {'Model':<14} {'MAP':<9} {'MRR':<9} {'P@1':<9} {'P@5':<9} {'P@10':<9}"
        print(header)
        print(f"  {'-'*64}")

        def row(label, r, tag=""):
            return (f"  {label:<14} {r['MAP']:<9.4f} {r['MRR']:<9.4f} "
                    f"{r['P@1']:<9.4f} {r['P@5']:<9.4f} {r['P@10']:<9.4f}  {tag}")

        print(row(ckpt_labels[best_map_ckpt],  bm,  "<-- Best MAP"))
        if best_mrr_ckpt != best_map_ckpt:
            print(row(ckpt_labels[best_mrr_ckpt],  bmr, "<-- Best MRR"))
        if best_both_ckpt not in (best_map_ckpt, best_mrr_ckpt):
            print(row(ckpt_labels[best_both_ckpt], bb,  "<-- Best MAP+MRR"))

    # ---- Full per-checkpoint table (highest topk) ----
    k_max = max(topk_values)
    print(f"\n\n{'='*70}")
    print(f"  Full sweep at TOP-K={k_max}")
    print(f"{'='*70}")
    header = f"  {'Checkpoint':<16} {'MAP':<9} {'MRR':<9} {'P@1':<9} {'P@5':<9} {'P@10':<9}"
    print(header)
    print(f"  {'-'*66}")
    for ckpt in ckpts:
        r = all_results[ckpt][k_max]
        label = ckpt_labels[ckpt]
        print(f"  {label:<16} {r['MAP']:<9.4f} {r['MRR']:<9.4f} "
              f"{r['P@1']:<9.4f} {r['P@5']:<9.4f} {r['P@10']:<9.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
