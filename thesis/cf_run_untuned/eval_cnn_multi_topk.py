import os
import json
import glob
import torch
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
from model_cnn import SimpleCNN

# --- Configuration (MUST MATCH TRAINING) ---
TARGET_QUERY_LEN = 32
TARGET_DOC_LEN = 225
PADDING_VALUE = 0.0
NORMALIZE_MATRICES = False
# --- End Configuration ---

def print_message(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def pad_or_truncate_tensor(tensor, target_shape, padding_value):
    """Pads or truncates a 2D tensor to target shape."""
    target_height, target_width = target_shape
    tensor = tensor[:target_height, :target_width]
    current_shape = tensor.shape
    pad_bottom = max(0, target_height - current_shape[0])
    pad_right = max(0, target_width - current_shape[1])
    if pad_bottom > 0 or pad_right > 0:
        tensor = torch.nn.functional.pad(
            tensor,
            (0, pad_right, 0, pad_bottom),
            mode='constant',
            value=padding_value
        )
    return tensor

def load_qrels_from_jsonl(jsonl_path):
    """Loads qrels from JSONL file (handles both 2-item and 3-item lines)."""
    qrels = defaultdict(set)
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if len(data) >= 2:
                    qid = int(data[0])
                    pos_id = int(data[1])
                    qrels[qid].add(pos_id)
            except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                continue
    print_message(f"Loaded qrels for {len(qrels)} queries from {jsonl_path}")
    return qrels

def load_qrels_from_txt(txt_path):
    """Loads qrels from Relevant.txt format."""
    qrels = {}
    with open(txt_path, 'r') as f:
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

def score_query_docs(qid, model, device, mats_dir):
    """Scores all available documents for a query."""
    scores = []
    target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)
    pattern = os.path.join(mats_dir, f"q{qid}_d*.pt")
    for matrix_path in glob.glob(pattern):
        try:
            fname = os.path.basename(matrix_path)
            did = int(fname.split('_d')[1].split('.pt')[0])
            mat = torch.load(matrix_path, map_location="cpu")

            # pad/truncate and normalize

            if NORMALIZE_MATRICES:
                m, s = mat.mean(), mat.std()
                if s > 1e-8:
                    mat = (mat - m) / s

            # reshape to [1,1,H,W]
            mat = mat.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(mat).item()
            scores.append((did, score))
        except Exception:
            continue
    return scores

def average_precision(retrieved, relevant, k=None):
    """Calculate Average Precision with optional top-k cutoff"""
    if not relevant:
        return 0.0
    k = k or len(retrieved)
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i
    return sum_precisions / len(relevant)

def reciprocal_rank(retrieved, relevant, k=None):
    """Calculate Reciprocal Rank with optional top-k cutoff"""
    k = k or len(retrieved)
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved, relevant, k):
    """Calculate Precision@K (always uses k as denominator)"""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)
    return relevant_in_top_k / k

def main():
    parser = argparse.ArgumentParser(description="Robust CNN Re-ranker Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to trained .pt model")
    parser.add_argument("--qrels_path", required=True, help="Qrels file path")
    parser.add_argument("--mats_dir", required=True, help="Directory with interaction matrices")
    parser.add_argument("--topk", type=str, default="5,10,20,50,100", help="Comma-separated TOPK values")
    parser.add_argument("--qrels_format", choices=["jsonl", "txt"], default="jsonl",
                        help="Qrels format: 'jsonl' for pairs/triplets, 'txt' for Relevant.txt")
    args = parser.parse_args()

    topk_values = sorted(set(int(k) for k in args.topk.split(',')))
    if not topk_values:
        raise ValueError("Invalid TOPK values provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Using device: {device}")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print_message(f"Loaded model from {args.model_path}")

    if args.qrels_format == "jsonl":
        qrels = load_qrels_from_jsonl(args.qrels_path)
    else:
        qrels = load_qrels_from_txt(args.qrels_path)

    available_qids = set()
    for path in glob.glob(os.path.join(args.mats_dir, "q*_d*.pt")):
        try:
            qid_str = os.path.basename(path).split("_d")[0][1:]
            available_qids.add(int(qid_str))
        except Exception:
            continue
    eval_qids = sorted(set(qrels.keys()) & available_qids)
    if not eval_qids:
        raise RuntimeError("No matching queries between qrels and matrix files")
    print_message(f"Evaluating on {len(eval_qids)} queries")

    print_message("Computing document scores...")
    rankings = {}
    for qid in tqdm(eval_qids, desc="Processing queries"):
        scored = score_query_docs(qid, model, device, args.mats_dir)
        scored.sort(key=lambda x: x[1], reverse=True)
        rankings[qid] = [did for did, _ in scored]

    print_message("Calculating metrics...")
    header = f"{'TOPK':<6} {'Queries':<8} {'MAP':<8} {'MRR':<8} {'P@1':<8} {'P@5':<8} {'P@10':<8}" 
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for k in topk_values:
        map_sum = mrr_sum = p1_sum = p5_sum = p10_sum = 0.0
        for qid in eval_qids:
            ret = rankings[qid]
            rel = qrels.get(qid, set())
            map_sum += average_precision(ret, rel, k)
            mrr_sum += reciprocal_rank(ret, rel, k)
            p1_sum += precision_at_k(ret, rel, 1)
            p5_sum += precision_at_k(ret, rel, 5)
            p10_sum += precision_at_k(ret, rel, 10)
        n = len(eval_qids)
        print(f"{k:<6} {n:<8} {map_sum/n:<8.4f} {mrr_sum/n:<8.4f} {p1_sum/n:<8.4f} {p5_sum/n:<8.4f} {p10_sum/n:<8.4f}")
    print(sep)

if __name__ == "__main__":
    main()
 #python thesis/cf_run_untuned/eval_cnn_multi_topk.py   --model_path thesis/cf_run_untuned/cnn_classifier_best_triliza_sm.pt   --qrels_path thesis/cf_run_untuned/test_triplets_cf_sm.jsonl   --mats_dir qd_matrices_cf_untuned/   --topk 5,10,20,25,40,50,80,100,150,200,250,400,500,1000  --qrels_format jsonl

 #python thesis/cf_run_untuned/eval_cnn_multi_topk.py    --model_path thesis/cf_run_untuned/recent_models_usurum/cnn_epoch_44.pt     --qrels_path CF_DataSet/Relevant.txt    --mats_dir qd_matrices_cf_untuned/  --topk 5,10,20,50,100,250,500,1000   --qrels_format txts