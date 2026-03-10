"""
Evaluate IVF candidates as a baseline (no CNN re-ranking).

Treats the retrieval order from IVF candidate files as the ranking and
computes MAP, MRR, P@1, P@5, P@10 at multiple top-k cutoffs.

Usage:
  python paper2/cfrun/eval_ivf_baseline.py \
    --qrels_path  paper2/cfrun/test_triplets.jsonl \
    --candidates_dir ivf_candidates_scifact \
    --topk 5,10,20,50,100,1000
"""
import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm


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


# ---------------------------------------------------------------------------
# Metrics
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
    print(f"\n{sep}\n{header}\n{sep}")

    for k in topk_values:
        map_sum = mrr_sum = p1_sum = p5_sum = p10_sum = 0.0
        for qid in eval_qids:
            ret = rankings[qid]
            rel = qrels[qid]
            map_sum  += average_precision(ret, rel, k)
            mrr_sum  += reciprocal_rank(ret, rel, k)
            p1_sum   += precision_at_k(ret, rel, 1)
            p5_sum   += precision_at_k(ret, rel, 5)
            p10_sum  += precision_at_k(ret, rel, 10)
        n = len(eval_qids)
        print(f"{k:<6} {n:<8} {map_sum/n:<8.4f} {mrr_sum/n:<8.4f} {p1_sum/n:<8.4f} {p5_sum/n:<8.4f} {p10_sum/n:<8.4f}")

    print(sep)


if __name__ == "__main__":
    main()
