"""
export_qd_matrices.py  (paper2/matrices_proce)
===============================================
Exports ColBERT query-document token-level similarity matrices to a single HDF5
database per query, using float16 storage.

Key improvements over paper/similarity_matrices_proce/export_qd_matrixes.py:
  1. IVF-first: only exports matrices for documents returned by IVF Stage 1
     candidates, not the full corpus (reduces files by 5-10x).
  2. HDF5 storage: all matrices for one query stored in one file
     q{qid}/matrices.h5  with datasets named  "d{did}".
  3. float16 storage: halves disk usage; converted to float32 at read time.
  4. Same export logic for non-IVF (full-corpus) mode — use --no_ivf flag.

Usage:
  # IVF-filtered (recommended):
  python paper2/matrices_proce/export_qd_matrices.py \
      --index ./experiments/CF19/indexes/CF19 \
      --output_dir ./qd_matrices_CF19_largetuned \
      --queries ./CF_DataSet/Queries.tsv \
      --candidates_dir ./ivf_candidates_cf19_largetuned \
      --save_heatmaps \
      --batch_size 500

  # Full-corpus (legacy behaviour):
  python paper2/matrices_proce/export_qd_matrices.py \
      --index /path/to/experiments/FACT1/indexes/FACT1 \
      --output_dir ./qd_matrices_scifact_paper2_full \
      --queries ./scifact_colbert_format/Queries.tsv \
      --no_ivf \
      --batch_size 500
"""

import os
import sys
import argparse
import json
import math
import gc

# Set GCC version BEFORE importing torch/ColBERT
os.environ['CC'] = '/usr/bin/gcc-11'
os.environ['CXX'] = '/usr/bin/g++-11'
os.environ['CUDAHOSTCXX'] = '/usr/bin/g++-11'

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

import torch
import h5py
import matplotlib.pyplot as plt

from colbert.search.index_storage import IndexScorer
from colbert.infra.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import print_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(max(size_bytes, 1), 1024)))
    i = min(i, len(size_name) - 1)
    return f"{round(size_bytes / math.pow(1024, i), 2)} {size_name[i]}"


def load_ivf_candidates(candidates_dir, qid):
    """
    Load IVF Stage-1 candidate PIDs for a given query from text file.
    Returns a list of int PIDs, or None if the file does not exist.
    """
    candidate_file = os.path.join(candidates_dir, f"q{qid}", f"q{qid}_ivf_candidates.txt")
    if not os.path.exists(candidate_file):
        return None
    pids = []
    with open(candidate_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pids.append(int(line))
                except ValueError:
                    continue
    return pids


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_colbert_similarity_matrices(
    index_dir,
    output_dir,
    queries,
    candidates_dir=None,   # None → full-corpus mode
    save_heatmaps=True,
    batch_size=100,
):
    """
    For each query, compute token-level similarity matrices D @ Q.T for the
    target document set (IVF candidates or full corpus) and save them into a
    per-query HDF5 file in float16.

    Output layout:
        output_dir/
            q0/
                matrices.h5      ← datasets: "d0", "d42", "d1337", ...
                q0_top_doc{pid}.png  (if save_heatmaps)
            q1/
                matrices.h5
            ...
            results_summary.json
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Resolve checkpoint path ----
    config = ColBERTConfig.load_from_index(index_dir)
    checkpoint_path_from_metadata = config.checkpoint
    effective_config_root = os.path.join(os.getcwd(), "colbert_run")

    relative = checkpoint_path_from_metadata.lstrip('./')
    absolute_ckpt = os.path.abspath(os.path.join(effective_config_root, relative))

    if os.path.isdir(absolute_ckpt):
        ckpt_path = absolute_ckpt
        print_message(f"Using local checkpoint: {ckpt_path}")
    else:
        ckpt_path = checkpoint_path_from_metadata   # HuggingFace ID
        print_message(f"Using remote HF checkpoint ID: {ckpt_path}")

    config.checkpoint = ckpt_path

    # ---- Load index scorer and checkpoint ----
    scorer = IndexScorer(index_dir, use_gpu=torch.cuda.is_available())
    scorer.set_embeddings_strided()
    checkpoint = Checkpoint(ckpt_path, config)

    num_all_docs = len(scorer.doclens)
    print_message(f"Index contains {num_all_docs} documents.")
    print_message(f"Export mode: {'IVF-filtered' if candidates_dir else 'FULL CORPUS'}")
    print_message(f"Batch size: {batch_size}")

    results = {}

    for q_idx, qtext in enumerate(queries):
        print_message(f"Query {q_idx + 1}/{len(queries)}: '{qtext[:60]}...'")

        # ---- Decide which PIDs to export ----
        if candidates_dir is not None:
            target_pids = load_ivf_candidates(candidates_dir, q_idx)
            if target_pids is None:
                print_message(f"  WARNING: No IVF candidate file for q{q_idx}. Skipping.")
                continue
            if len(target_pids) == 0:
                print_message(f"  WARNING: Empty candidate list for q{q_idx}. Skipping.")
                continue
        else:
            target_pids = list(range(num_all_docs))

        # ---- Encode query ----
        Q = checkpoint.queryFromText([qtext], bsize=1, to_cpu=False)  # [1, q_len, dim]
        Q_for_matmul = Q.squeeze(0).transpose(0, 1)                   # [dim, q_len]

        # ---- Create per-query subdirectory and HDF5 file ----
        query_subdir = os.path.join(output_dir, f"q{q_idx}")
        os.makedirs(query_subdir, exist_ok=True)
        h5_path = os.path.join(query_subdir, "matrices.h5")

        best_score  = -float('inf')
        top_pid     = -1
        top_matrix  = None
        total_bytes = 0
        num_exported = 0

        with h5py.File(h5_path, 'w') as h5f:
            # Process in batches for GPU efficiency
            for batch_start in range(0, len(target_pids), batch_size):
                batch_pids = target_pids[batch_start:batch_start + batch_size]

                if batch_start % 1000 == 0 and batch_start > 0:
                    print_message(f"  q{q_idx}: {batch_start}/{len(target_pids)} docs processed...")

                # Batch lookup
                D_packed_batch, D_lens_batch = scorer.lookup_pids(batch_pids)
                if D_packed_batch is None or D_packed_batch.size(0) == 0:
                    continue

                if D_packed_batch.dtype == torch.float16:
                    D_packed_batch = D_packed_batch.to(torch.float32)

                # Batch matrix multiplication: [total_doc_tokens, q_len]
                Q_dev = Q_for_matmul.to(D_packed_batch.device)
                sim_batch = D_packed_batch @ Q_dev

                # Split back to individual documents
                offset = 0
                for i, pid in enumerate(batch_pids):
                    doc_len = D_lens_batch[i]
                    sim_doc = sim_batch[offset:offset + doc_len]  # [doc_tokens, q_len]
                    offset += doc_len

                    if sim_doc.numel() == 0:
                        continue

                    # Shape: [q_len, doc_tokens]  (rows = query tokens, cols = doc tokens)
                    mat = sim_doc.transpose(0, 1).cpu()

                    maxsim = mat.max().item()

                    # --- Save as float16 in HDF5 ---
                    mat_fp16 = mat.to(torch.float16).numpy()
                    dataset_name = f"d{pid}"
                    h5f.create_dataset(dataset_name, data=mat_fp16,
                                       compression="gzip", compression_opts=4)

                    num_exported += 1

                    if maxsim > best_score:
                        best_score = maxsim
                        top_pid    = pid
                        top_matrix = mat.clone()   # keep float32 for heatmap

        total_bytes = os.path.getsize(h5_path) if os.path.exists(h5_path) else 0
        print_message(f"  Exported {num_exported} matrices → {h5_path} "
                      f"({format_size(total_bytes)})")

        # ---- Optional heatmap for top-scoring document ----
        if save_heatmaps and top_matrix is not None and top_pid != -1:
            png_path = os.path.join(output_dir, f"q{q_idx}_top_doc{top_pid}.png")
            try:
                plt.figure(figsize=(5, 4))
                plt.imshow(top_matrix.numpy(), aspect="auto", cmap="viridis",
                           vmin=0.0, vmax=1.0)
                plt.colorbar()
                plt.xlabel(f"Top Doc (PID {top_pid}) token position")
                plt.ylabel(f"Query {q_idx} token position")
                plt.title(f"Q{q_idx} vs Doc {top_pid} (MaxSim: {best_score:.4f})")
                plt.savefig(png_path, dpi=100, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print_message(f"  Error saving heatmap: {e}")

        results[q_idx] = {
            "query_text_snippet"  : qtext[:100] + "..." if len(qtext) > 100 else qtext,
            "num_docs_exported"   : num_exported,
            "top_scoring_pid"     : top_pid,
            "best_maxsim_score"   : best_score,
            "h5_file_bytes"       : total_bytes,
            "mode"                : "ivf_filtered" if candidates_dir else "full_corpus",
        }

        # Memory cleanup after each query
        del Q, Q_for_matmul
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    summary_path = os.path.join(output_dir, "results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print_message(f"Summary saved to {summary_path}")
    print_message(f"Done. All HDF5 files written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export ColBERT QxD similarity matrices to per-query HDF5 files (float16)."
    )
    parser.add_argument("--index",         required=True,
                        help="Path to ColBERT index directory.")
    parser.add_argument("--output_dir",    required=True,
                        help="Directory to save per-query HDF5 files.")
    parser.add_argument("--queries",       required=True,
                        help="TSV file with queries (qid<TAB>text or just text per line).")
    parser.add_argument("--candidates_dir", default=None,
                        help="IVF candidates root dir (ivf_candidates_scifact). "
                             "If omitted → full-corpus export.")
    parser.add_argument("--no_ivf",        action="store_true",
                        help="Force full-corpus export even if --candidates_dir is provided.")
    parser.add_argument("--save_heatmaps", action="store_true",
                        help="Save a PNG heatmap for the top-scoring document per query.")
    parser.add_argument("--batch_size",    type=int, default=100,
                        help="Documents per GPU batch (default: 100, try 300-500 for speed).")

    args = parser.parse_args()

    # Resolve candidates_dir
    candidates_dir = None
    if args.candidates_dir and not args.no_ivf:
        if os.path.isdir(args.candidates_dir):
            candidates_dir = args.candidates_dir
            print_message(f"IVF mode: candidates from {candidates_dir}")
        else:
            print_message(f"WARNING: --candidates_dir '{args.candidates_dir}' not found. "
                          f"Falling back to full-corpus export.")

    # Load queries
    queries_input = []
    try:
        with open(args.queries, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if not parts or not parts[0].strip():
                    continue
                query_text = parts[1] if len(parts) > 1 else parts[0]
                queries_input.append(query_text.strip())
        if not queries_input:
            print_message("Error: No valid queries found. Exiting.")
            sys.exit(1)
    except FileNotFoundError:
        print_message(f"Error: Queries file not found at {args.queries}. Exiting.")
        sys.exit(1)

    export_colbert_similarity_matrices(
        index_dir      = args.index,
        output_dir     = args.output_dir,
        queries        = queries_input,
        candidates_dir = candidates_dir,
        save_heatmaps  = args.save_heatmaps,
        batch_size     = args.batch_size,
    )
