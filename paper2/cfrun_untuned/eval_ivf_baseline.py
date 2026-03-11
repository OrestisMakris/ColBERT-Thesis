"""
Evaluate IVF candidates as a baseline (no CNN re-ranking).

Baseline 1 — IVF order : treats the retrieval order from IVF candidate files
             as the ranking.
Baseline 2 — MaxSim    : re-ranks IVF candidates by ColBERT MaxSim score
             (sum of per-query-token max similarity over doc tokens) computed
             from the pre-exported HDF5 matrices.

Computes MAP, MRR, P@1, P@5, P@10 at multiple top-k cutoffs for both baselines.

Usage:
  python paper2/cfrun_untuned/eval_ivf_baseline.py \
    --qrels_path     paper2/cfrun_untuned/test_triplets_hard.jsonl \
    --candidates_dir ivf_candidates_cf17_untuned \
    --matrices_dir   qd_matrices_CF17_untuned \
    --topk 5,10,15,20,25,50,100,250,500,1000
"""
import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import h5py
import numpy as np


def load_qrels(jsonl_path: str) -> dict:
    """Load ground-truth qrels from JSONL (handles 2- and 3-item lines)."""
    qrels = defaultdict(set)
    with open(jsonl_path) as f:
        for line in f:
            try:
                data = json.loads(line)
                if len(data) >= 2:
                    qrels[int(data[0])].add(int(data[1]))
            except Exception:
                continue
    print(f"Loaded qrels for {len(qrels)} queries.")
    return qrels


def load_rankings_from_candidates(candidates_dir: str) -> dict:
    """Read IVF candidate files and treat their order as the ranking."""
    rankings = {}
    query_dirs = [
        d for d in os.listdir(candidates_dir)
        if d.startswith("q") and os.path.isdir(os.path.join(candidates_dir, d))
    ]
    print(f"Found {len(query_dirs)} query directories in {candidates_dir}")

    for q_dir in tqdm(query_dirs, desc="Loading candidates"):
        try:
            qid       = int(q_dir[1:])
            file_path = os.path.join(candidates_dir, q_dir, f"{q_dir}_ivf_candidates.txt")
            if not os.path.exists(file_path):
                continue
            docs = []
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(int(line))
            rankings[qid] = docs
        except ValueError:
            continue

    return rankings


def load_rankings_from_maxsim(candidates_dir: str, matrices_dir: str) -> dict:
    """
    Re-rank IVF candidates by ColBERT MaxSim score using pre-exported HDF5 matrices.
    MaxSim(q, d) = sum over query tokens of max over doc tokens of sim(q_i, d_j).
    Returns rankings sorted by descending MaxSim score.
    """
    rankings = {}
    query_dirs = [
        d for d in os.listdir(candidates_dir)
        if d.startswith("q") and os.path.isdir(os.path.join(candidates_dir, d))
    ]

    for q_dir in tqdm(query_dirs, desc="MaxSim ranking"):
        try:
            qid = int(q_dir[1:])
        except ValueError:
            continue

        h5_path = os.path.join(matrices_dir, f"q{qid}", "matrices.h5")
        if not os.path.exists(h5_path):
            continue

        scores = {}
        with h5py.File(h5_path, "r") as h5f:
            for key in h5f.keys():
                mat = h5f[key][:].astype(np.float32)  # [q_len, doc_tokens]
                scores[int(key[1:])] = float(mat.max(axis=1).sum())

        if scores:
            rankings[qid] = sorted(scores, key=lambda d: scores[d], reverse=True)

    print(f"MaxSim rankings computed for {len(rankings)} queries.")
    return rankings



# ---------------------------------------------------------------------------

def average_precision(retrieved, relevant, k=None):
    if not relevant:
        return 0.0
    k = k or len(retrieved)
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant)


def reciprocal_rank(retrieved, relevant, k=None):
    k = k or len(retrieved)
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved, relevant, k):
    if k <= 0:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in relevant) / k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate IVF Candidates Baseline (No CNN)")
    parser.add_argument("--qrels_path",     required=True, help="Path to test_triplets.jsonl")
    parser.add_argument("--candidates_dir", required=True, help="Path to IVF candidates directory")
    parser.add_argument("--matrices_dir",   default=None,  help="Path to HDF5 matrices directory for MaxSim baseline (optional)")
    parser.add_argument("--topk",           type=str, default="5,10,20,50,100,1000",
                        help="Comma-separated K values")
    args = parser.parse_args()

    qrels    = load_qrels(args.qrels_path)
    rankings = load_rankings_from_candidates(args.candidates_dir)

    eval_qids = sorted(set(qrels.keys()) & set(rankings.keys()))
    print(f"Evaluating on {len(eval_qids)} common queries.")

    topk_values = [int(k) for k in args.topk.split(",")]
    header = f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}"
    sep    = "=" * len(header)

    def print_results(label, rank_dict):
        print(f"\n--- {label} ---")
        print(f"{sep}\n{header}\n{sep}")
        for k in topk_values:
            map_sum = mrr_sum = p1_sum = p5_sum = p10_sum = 0.0
            for qid in eval_qids:
                ret = rank_dict[qid]
                rel = qrels[qid]
                map_sum  += average_precision(ret, rel, k)
                mrr_sum  += reciprocal_rank(ret, rel, k)
                p1_sum   += precision_at_k(ret, rel, 1)
                p5_sum   += precision_at_k(ret, rel, 5)
                p10_sum  += precision_at_k(ret, rel, 10)
            n = len(eval_qids)
            print(f"{k:<6} {n:<8} {map_sum/n:<8.4f} {mrr_sum/n:<8.4f} {p1_sum/n:<8.4f} {p5_sum/n:<8.4f} {p10_sum/n:<8.4f}")
        print(sep)

    print_results("Baseline 1: IVF order", rankings)

    if args.matrices_dir:
        maxsim_rankings = load_rankings_from_maxsim(args.candidates_dir, args.matrices_dir)
        maxsim_eval_qids = sorted(set(qrels.keys()) & set(maxsim_rankings.keys()))
        if maxsim_eval_qids:
            print_results("Baseline 2: MaxSim re-ranking", maxsim_rankings)
        else:
            print("No queries with MaxSim rankings found — check --matrices_dir path.")
    else:
        print("\n(Skipping MaxSim baseline — pass --matrices_dir to enable it.)")


if __name__ == "__main__":
    main()
