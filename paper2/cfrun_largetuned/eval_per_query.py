"""
Per-query evaluation of the SimpleCNN re-ranker.

Outputs MAP, MRR, P@1, P@5, P@10 for every individual test query,
plus aggregate totals, at topk = 5,10,20,50,100,250,500,1000.

Usage:
  python paper2/cfrun/eval_per_query.py \
      --model_path paper2/cfrun/recent_models_arcade_very_nice/cnn_epoch_72.pt \
      --qrels_path paper2/cfrun/test_triplets_hard.jsonl \
      --mats_dir qd_matrices_CF17_untuned/ \
      --candidates_dir ivf_candidates_cf17_untuned/ \
      --topk 5,10,20,50,100,250,500,1000 \
      --batch_size 500
"""
import os
import json
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
TARGET_DOC_LEN     = 140
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-query CNN re-ranker evaluation"
    )
    parser.add_argument("--model_path",     required=True)
    parser.add_argument("--qrels_path",     required=True)
    parser.add_argument("--mats_dir",       required=True)
    parser.add_argument("--candidates_dir", default=None)
    parser.add_argument("--topk",           default="5,10,20,50,100,250,500,1000")
    parser.add_argument("--batch_size",     type=int, default=128)
    args = parser.parse_args()

    topk_values = sorted(set(int(k) for k in args.topk.split(",")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Device: {device}  |  Model: {args.model_path}")

    qrels = load_qrels(args.qrels_path)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

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

    # ---- Score all queries ----
    rankings = {}
    for qid in tqdm(eval_qids, desc="Scoring queries"):
        rankings[qid] = score_query(
            qid, model, device, args.mats_dir, args.batch_size, args.candidates_dir
        )

    # ---- Per-query table for each top-k ----
    for k in topk_values:
        print(f"\n{'='*76}")
        print(f"  PER-QUERY RESULTS  @  TOP-K = {k}")
        print(f"{'='*76}")
        header = f"  {'QID':<8} {'#Rel':<6} {'MAP':<9} {'MRR':<9} {'P@1':<9} {'P@5':<9} {'P@10':<9}"
        print(header)
        print(f"  {'-'*72}")

        total_map = total_mrr = total_p1 = total_p5 = total_p10 = 0.0

        per_query_rows = []
        for qid in eval_qids:
            ret = rankings[qid]
            rel = qrels.get(qid, set())
            ap  = average_precision(ret, rel, k)
            rr  = reciprocal_rank(ret, rel, k)
            p1  = precision_at_k(ret, rel, 1)
            p5  = precision_at_k(ret, rel, 5)
            p10 = precision_at_k(ret, rel, 10)

            total_map  += ap
            total_mrr  += rr
            total_p1   += p1
            total_p5   += p5
            total_p10  += p10

            per_query_rows.append((qid, len(rel), ap, rr, p1, p5, p10))

        for qid, n_rel, ap, rr, p1, p5, p10 in per_query_rows:
            print(f"  {qid:<8} {n_rel:<6} {ap:<9.4f} {rr:<9.4f} {p1:<9.4f} {p5:<9.4f} {p10:<9.4f}")

        n = len(eval_qids)
        print(f"  {'-'*72}")
        print(f"  {'MEAN':<8} {'':<6} "
              f"{total_map/n:<9.4f} {total_mrr/n:<9.4f} "
              f"{total_p1/n:<9.4f} {total_p5/n:<9.4f} {total_p10/n:<9.4f}")
        print(f"{'='*76}")

        # ---- Best and worst queries at this top-k ----
        sorted_by_map = sorted(per_query_rows, key=lambda r: r[2], reverse=True)
        print(f"\n  Top-5 queries by MAP@{k}:")
        for qid, n_rel, ap, rr, p1, p5, p10 in sorted_by_map[:5]:
            print(f"    q{qid:<6} MAP={ap:.4f}  MRR={rr:.4f}  #rel={n_rel}")

        print(f"\n  Worst-5 queries by MAP@{k}:")
        for qid, n_rel, ap, rr, p1, p5, p10 in sorted_by_map[-5:]:
            print(f"    q{qid:<6} MAP={ap:.4f}  MRR={rr:.4f}  #rel={n_rel}")


if __name__ == "__main__":
    main()
