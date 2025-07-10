import os
import json
import glob
import torch
import argparse
import time
from tqdm import tqdm
from collections import defaultdict
import numpy as np
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
    """Loads qrels from JSONL file."""
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

def score_query_docs(qid, model, device, mats_dir):
    """Scores all available documents for a query."""
    scores = []
    target_shape = (TARGET_QUERY_LEN, TARGET_DOC_LEN)
    pattern = os.path.join(mats_dir, f"q{qid}_d*.pt")
    matrix_files = glob.glob(pattern)
    
    if not matrix_files:
        return [], 0 # Return 0 documents processed

    for matrix_path in matrix_files:
        try:
            fname = os.path.basename(matrix_path)
            did = int(fname.split('_d')[1].split('.pt')[0])
            mat = torch.load(matrix_path, map_location="cpu")

            if NORMALIZE_MATRICES:
                m, s = mat.mean(), mat.std()
                if s > 1e-8:
                    mat = (mat - m) / s
            
            mat = pad_or_truncate_tensor(mat, target_shape, PADDING_VALUE)
            mat = mat.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(mat).item()
            scores.append((did, score))
        except Exception:
            continue
    return scores, len(matrix_files)

def main():
    parser = argparse.ArgumentParser(description="Time CNN Re-ranker Retrieval")
    parser.add_argument("--model_path", required=True, help="Path to trained .pt model")
    parser.add_argument("--qrels_path", required=True, help="Qrels file path for selecting queries")
    parser.add_argument("--mats_dir", required=True, help="Directory with interaction matrices")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_message(f"Using device: {device}")

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print_message(f"Loaded model from {args.model_path}")

    qrels = load_qrels_from_jsonl(args.qrels_path)

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
    print_message(f"Found {len(eval_qids)} queries to evaluate for timing.")

    print_message("Starting timing evaluation...")
    query_times = []
    total_docs_processed = 0

    for qid in tqdm(eval_qids, desc="Timing queries"):
        start_time = time.time()
        scored_docs, num_docs = score_query_docs(qid, model, device, args.mats_dir)
        end_time = time.time()
        
        if num_docs > 0:
            elapsed_time = end_time - start_time
            query_times.append(elapsed_time)
            total_docs_processed += num_docs

    print_message("Timing evaluation complete.")
    
    # --- Print Timing Statistics ---
    if query_times:
        total_time = sum(query_times)
        num_queries = len(query_times)
        avg_time_per_query = np.mean(query_times)
        std_time_per_query = np.std(query_times)
        
        header = "Timing Results"
        sep = "=" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        print(f"Total queries timed: {num_queries}")
        print(f"Total documents processed: {total_docs_processed}")
        print(f"Total time elapsed: {total_time:.4f} seconds")
        print(f"Average time per query: {avg_time_per_query:.4f} seconds")
        print(f"Std dev of query time: {std_time_per_query:.4f} seconds")
        if total_docs_processed > 0:
            avg_time_per_doc = total_time / total_docs_processed
            print(f"Average time per document score: {avg_time_per_doc * 1000:.4f} ms")
        print(sep)
    else:
        print_message("No queries were timed.")

if __name__ == "__main__":
    main()