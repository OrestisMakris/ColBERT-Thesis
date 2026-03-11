"""
Evaluate the trained SimpleCNN re-ranker on HDF5 matrix files.

Key differences from paper/fact_run/eval_cnn_multi_topk.py:
  1. resize_tensor() via bilinear interpolation instead of pad/truncate
  2. Matrices loaded from HDF5 (q{qid}/matrices.h5, dataset "d{doc_id}")
  3. Candidates filtering reads IVF .txt files and cross-references HDF5 keys
  4. float16 numpy → float32 tensor conversion at load time
"""
import os
import json
import h5py
import torch
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
import torch.nn.functional as F

from model_cnn import SimpleCNN


# ---------------------------------------------------------------------------
# Configuration (must match training)
# ---------------------------------------------------------------------------
TARGET_QUERY_LEN  = 32
TARGET_DOC_LEN    = 128
NORMALIZE_MATRICES = False
BATCH_SIZE         = 128
# ---------------------------------------------------------------------------


def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def resize_tensor(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Bilinear resize of a 2-D float tensor to (target_height, target_width)."""
    return F.interpolate(
        tensor.unsqueeze(0).unsqueeze(0),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Qrels loading
# ---------------------------------------------------------------------------

def load_qrels_from_jsonl(jsonl_path: str) -> dict:
    qrels = defaultdict(set)
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if len(data) >= 2:
                    qrels[int(data[0])].add(int(data[1]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    print_message(f"Loaded qrels for {len(qrels)} queries from {jsonl_path}")
    return qrels


def load_qrels_from_txt(txt_path: str) -> dict:
    qrels = {}
    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            ids = set()
            for tok in line.strip().split():
                try:
                    ids.add(int(tok))
                except ValueError:
                    continue
            qrels[idx] = ids
    print_message(f"Loaded qrels for {len(qrels)} queries from {txt_path}")
    return qrels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_query_docs_batched(
    qid,
    model,
    device,
    mats_dir: str,
    batch_size: int = BATCH_SIZE,
    candidates_dir: str = None,
) -> list:
    """Score all (or candidate-filtered) docs for query qid.

    Returns list of (doc_id, score) sorted by score descending.
    """
    target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)
    h5_path = os.path.join(mats_dir, f"q{qid}", "matrices.h5")

    if not os.path.isfile(h5_path):
        return []

    # Determine which doc keys to score
    with h5py.File(h5_path, "r") as h5f:
        all_keys = set(h5f.keys())  # e.g. {"d0", "d42", ...}

    if candidates_dir:
        cand_file = os.path.join(candidates_dir, f"q{qid}", f"q{qid}_ivf_candidates.txt")
        if not os.path.exists(cand_file):
            return []
        doc_keys = []
        with open(cand_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    key = f"d{line}"
                    if key in all_keys:
                        doc_keys.append(key)
    else:
        doc_keys = sorted(all_keys, key=lambda k: int(k[1:]))

    if not doc_keys:
        return []

    all_scores = []

    with h5py.File(h5_path, "r") as h5f:
        for i in range(0, len(doc_keys), batch_size):
            batch_keys = doc_keys[i : i + batch_size]
            batch_mats = []
            batch_dids = []

            for key in batch_keys:
                try:
                    # float16 numpy → float32 tensor
                    mat = torch.from_numpy(h5f[key][:]).float()
                    mat = resize_tensor(mat, target_shape)
                    if NORMALIZE_MATRICES:
                        m, s = mat.mean(), mat.std()
                        if s > 1e-8:
                            mat = (mat - m) / s
                    batch_mats.append(mat)
                    batch_dids.append(int(key[1:]))  # "d42" → 42
                except Exception:
                    continue

            if not batch_mats:
                continue

            batch_tensor = torch.stack([m.unsqueeze(0) for m in batch_mats]).to(device)
            with torch.no_grad():
                scores = model(batch_tensor).squeeze().cpu()
            if scores.dim() == 0:
                scores = [scores.item()]
            else:
                scores = scores.tolist()
            all_scores.extend(zip(batch_dids, scores))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    return all_scores


def process_single_query(qid, model_path, device_str, mats_dir, batch_size, candidates_dir):
    """Worker function for multiprocessing — loads model fresh per process."""
    device = torch.device("cpu")
    model  = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scored = score_query_docs_batched(qid, model, device, mats_dir, batch_size, candidates_dir)
    return qid, [did for did, _ in scored]


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
    parser = argparse.ArgumentParser(
        description="CNN Re-ranker Evaluation — HDF5 matrices + bilinear resize (paper2)"
    )
    parser.add_argument("--model_path",    required=True, help="Path to trained .pt model")
    parser.add_argument("--qrels_path",    required=True, help="Qrels file path (jsonl or txt)")
    parser.add_argument("--mats_dir",      required=True, help="Directory with HDF5 matrices (paper2 format)")
    parser.add_argument("--topk",          type=str, default="5,10,20,50,100",
                        help="Comma-separated K values for evaluation")
    parser.add_argument("--qrels_format",  choices=["jsonl", "txt"], default="jsonl")
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--num_workers",   type=int, default=None)
    parser.add_argument("--use_multiprocessing", action="store_true",
                        help="Enable multiprocessing (CPU only)")
    parser.add_argument("--candidates_dir", type=str, default=None,
                        help="IVF candidates directory; only those docs are scored")
    args = parser.parse_args()

    topk_values = sorted(set(int(k) for k in args.topk.split(",")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Device: {device}  |  Batch size: {args.batch_size}")

    if args.qrels_format == "jsonl":
        qrels = load_qrels_from_jsonl(args.qrels_path)
    else:
        qrels = load_qrels_from_txt(args.qrels_path)

    # Discover available query directories
    available_qids = set()
    for name in os.listdir(args.mats_dir):
        if name.startswith("q") and os.path.isdir(os.path.join(args.mats_dir, name)):
            try:
                available_qids.add(int(name[1:]))
            except ValueError:
                continue

    eval_qids = sorted(set(qrels.keys()) & available_qids)
    if not eval_qids:
        raise RuntimeError("No matching queries between qrels and matrix files.")
    print_message(f"Evaluating on {len(eval_qids)} queries.")

    if args.use_multiprocessing:
        num_workers = args.num_workers or min(cpu_count(), len(eval_qids))
        print_message(f"Multiprocessing: {num_workers} workers (CPU).")
        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        process_func = partial(
            process_single_query,
            model_path=args.model_path,
            device_str="cpu",
            mats_dir=args.mats_dir,
            batch_size=args.batch_size,
            candidates_dir=args.candidates_dir,
        )
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, eval_qids),
                total=len(eval_qids),
                desc="Processing queries",
            ))
        rankings = dict(results)

    else:
        print_message("Sequential batched inference (GPU-enabled).")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model.eval()

        rankings = {}
        for qid in tqdm(eval_qids, desc="Processing queries"):
            scored = score_query_docs_batched(
                qid, model, device, args.mats_dir, args.batch_size, args.candidates_dir
            )
            rankings[qid] = [did for did, _ in scored]

    # ---- Print metrics table ----
    print_message("Calculating metrics...")
    header = f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}"
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for k in topk_values:
        map_sum = mrr_sum = p1_sum = p5_sum = p10_sum = 0.0
        for qid in eval_qids:
            ret = rankings[qid]
            rel = qrels.get(qid, set())
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

# ---------------------------------------------------------------------------
# Example usage:
#
# Sequential (GPU):
#   python paper2/cfrun/eval_cnn_multi_topk.py \
#     --model_path paper2/cfrun/recent_models_triliza/cnn_epoch_41.pt \
#     --qrels_path paper2/cfrun/test_triplets_hard.jsonl \
#     --mats_dir qd_matrices_CF17_untuned/ \
#     --candidates_dir ivf_candidates_cf17_untuned/ \
#     --topk 5,10,20,50,100,250,500,1000 \
#     --batch_size 500
#
# Multiprocessing (CPU):
#   python paper2/cfrun/eval_cnn_multi_topk.py \
#     --model_path paper2/cfrun/cnn_classifier_ldr.pt \
#     --qrels_path paper2/cfrun/test_triplets.jsonl \
#     --mats_dir qd_matrices_scifact_paper2/ \
#     --topk 5,10,20,50,100 \
#     --batch_size 512 \
#     --use_multiprocessing \
#     --num_workers 8
# ---------------------------------------------------------------------------
