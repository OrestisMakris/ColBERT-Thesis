import os , sys

current_dir   = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

import torch
from colbert.search.index_storage import IndexScorer
from colbert.infra.config import ColBERTConfig

from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import print_message
import matplotlib.pyplot as plt
import json
import math
import argparse

def format_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    try:
        i = int(math.floor(math.log(size_bytes, 1024)))
        if i >= len(size_name): # Handle extremely large sizes beyond YB
            i = len(size_name) - 1
    except ValueError: # math domain error for log(0) or negative
        return "0B" if size_bytes == 0 else str(size_bytes) + "B" # Should not happen if first check is 0
        
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def export_colbert_similarity_matrices(
    index_dir, output_dir, queries, save_heatmaps=True
):
    """
    Extracts and exports token-level similarity matrices directly from ColBERT
    for ALL query-document pairs in the index.
    Generates one heatmap per query for the top-scoring document.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    config = ColBERTConfig.load_from_index(index_dir)
    checkpoint_path_from_metadata = config.checkpoint 
    project_root = os.getcwd() 
    # Assuming the script is run from ColBERT-Thesis, and 'experiments' is inside 'colbert_run'
    effective_config_root = os.path.join(project_root, "colbert_run")

    if checkpoint_path_from_metadata.startswith('./'):
        relative_checkpoint_path = checkpoint_path_from_metadata[2:]
    else:
        relative_checkpoint_path = checkpoint_path_from_metadata
    
    absolute_checkpoint_path = os.path.abspath(os.path.join(effective_config_root, relative_checkpoint_path))
    
    print_message(f"Original config.checkpoint from metadata: {checkpoint_path_from_metadata}")
    print_message(f"Resolved absolute_checkpoint_path: {absolute_checkpoint_path}")

    hf_id = checkpoint_path_from_metadata
    if os.path.isdir(absolute_checkpoint_path):
        ckpt_path = absolute_checkpoint_path
        print_message(f"Using local checkpoint: {ckpt_path}")
    else:
        ckpt_path = hf_id
        print_message(f"Using remote HF checkpoint ID: {ckpt_path}")

    config.checkpoint = ckpt_path

    scorer = IndexScorer(index_dir, use_gpu=torch.cuda.is_available())
    scorer.set_embeddings_strided()

    checkpoint = Checkpoint(ckpt_path, config)

    num_all_docs_in_index = len(scorer.doclens)
    print_message(f"Found {num_all_docs_in_index} documents in the index.")
        
    for q_idx, qtext in enumerate(queries):
        print_message(f"Processing query {q_idx + 1}/{len(queries)}: '{qtext[:50]}...'")
        
        Q = checkpoint.queryFromText([qtext], bsize=1, to_cpu=False) # Shape: [1, q_len, dim]
        
        query_results_pids = []
        query_results_scores = []
        
        num_matrices_for_this_query = 0
        total_pt_size_for_this_query = 0
        
        best_score_for_query = -float('inf')
        top_scoring_pid_for_query = -1
        top_scoring_matrix_for_query = None

        for pid in range(num_all_docs_in_index):
            if pid % 1000 == 0 and pid > 0 and num_all_docs_in_index > 1000: # Log progress for large sets
                 print_message(f"  Query {q_idx + 1}: Processing document {pid}/{num_all_docs_in_index}...")

            D_packed, D_lens = scorer.lookup_pids([pid]) 
            
            if D_packed is None or D_packed.size(0) == 0:
                print_message(f"  Warning: Could not retrieve embeddings for PID {pid} (Query {q_idx + 1}). Skipping.")
                continue

            if D_packed.dtype == torch.float16:
                D_packed = D_packed.to(torch.float32)

            Q_for_matmul = Q.squeeze(0).transpose(0, 1).to(D_packed.device)
            sim_flat_for_one_doc = D_packed @ Q_for_matmul # Shape: [doc_tokens, q_len]
            
            if sim_flat_for_one_doc.numel() > 0:
                max_sim_score = sim_flat_for_one_doc.max().item()
            else:
                max_sim_score = 0.0

            query_results_pids.append(pid)
            query_results_scores.append(max_sim_score)
            
            mat_unpadded = sim_flat_for_one_doc.transpose(0,1).cpu() # Shape [q_len, doc_tokens]
            
            filepath_pt = f"{output_dir}/q{q_idx}_d{pid}.pt"
            torch.save(mat_unpadded, filepath_pt)
            
            if os.path.exists(filepath_pt):
                total_pt_size_for_this_query += os.path.getsize(filepath_pt)
                num_matrices_for_this_query += 1
                # Per-matrix log removed for brevity as requested
                # print_message(f"  Saved matrix for q{q_idx}_d{pid} (Shape: {list(mat_unpadded.shape)}) to {filepath_pt}")

            if max_sim_score > best_score_for_query:
                best_score_for_query = max_sim_score
                top_scoring_pid_for_query = pid
                top_scoring_matrix_for_query = mat_unpadded.clone() 
        
        results[q_idx] = {
            "query_text_snippet": qtext[:100] + "..." if len(qtext) > 100 else qtext,
            "pids_processed_count": len(query_results_pids), 
            # "maxsim_scores": query_results_scores, # Can be very long, omitting from default JSON
            "top_scoring_pid": top_scoring_pid_for_query, 
            "best_maxsim_score": best_score_for_query
        }
        
        heatmap_log_info = ""
        if save_heatmaps and top_scoring_matrix_for_query is not None and top_scoring_pid_for_query != -1:
            filepath_png = f"{output_dir}/q{q_idx}_top_doc{top_scoring_pid_for_query}.png"
            try:
                plt.figure(figsize=(5, 4)) # Adjust as needed
                plt.imshow(top_scoring_matrix_for_query.numpy(), aspect="auto", cmap="viridis")
                plt.colorbar()
                plt.xlabel(f"Top Document (PID {top_scoring_pid_for_query}) token position")
                plt.ylabel(f"Query QID {q_idx} token position")
                plt.title(f"Query {q_idx} vs Top Doc {top_scoring_pid_for_query} (MaxSim: {best_score_for_query:.4f})")
                plt.savefig(filepath_png, dpi=150, bbox_inches="tight")
                plt.close() # Close the figure to free memory
                if os.path.exists(filepath_png):
                    png_size = os.path.getsize(filepath_png)
                    heatmap_log_info = (f" Exported 1 heatmap ({format_size(png_size)}) "
                                        f"for top doc {top_scoring_pid_for_query} "
                                        f"(Matrix Shape: {list(top_scoring_matrix_for_query.shape)}).")
            except Exception as e:
                print_message(f"  Error saving heatmap for query {q_idx}, top_doc {top_scoring_pid_for_query}: {e}")
                heatmap_log_info = " Error generating heatmap."


        pt_size_formatted = format_size(total_pt_size_for_this_query)
        log_message_for_query = (f"Query {q_idx + 1} processed: Exported {num_matrices_for_this_query} matrices "
                                 f"({pt_size_formatted}).{heatmap_log_info}")
        print_message(log_message_for_query)
    
    results_filepath = f"{output_dir}/results_summary.json"
    with open(results_filepath, "w") as f:
        json.dump(results, f, indent=2)
    print_message(f"Saved summary of processing to {results_filepath}")
    
    actual_exported_count = sum(r['pids_processed_count'] for r_idx, r in results.items() if 'pids_processed_count' in r)
    print_message(f"Completed. Exported {actual_exported_count} total matrices to {output_dir} (Processed {num_all_docs_in_index} docs for {len(queries)} queries).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ColBERT QxD similarity matrices for ALL query-document pairs. Optionally saves one heatmap per query for its top-scoring document.")
    parser.add_argument("--index", required=True, help="Path to ColBERT index directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save matrices and heatmaps.")
    parser.add_argument("--queries", required=True, help="Path to TSV file with queries (qid<tab>query or just query per line).")
    parser.add_argument("--save_heatmaps", action='store_true', help="Save a .png heatmap for the top-scoring document for each query.")
    
    args = parser.parse_args()
    
    queries_input = []
    try:
        with open(args.queries, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split('\t')
                if not parts or not parts[0].strip(): # Skip empty or whitespace-only lines
                    print_message(f"Warning: Skipping empty or invalid line {line_idx+1} in queries file.")
                    continue
                query_text = parts[1] if len(parts) > 1 else parts[0]
                queries_input.append(query_text.strip())
        if not queries_input:
            print_message("Error: No valid queries found in the input file. Exiting.")
            exit(1)
    except FileNotFoundError:
        print_message(f"Error: Queries file not found at {args.queries}. Exiting.")
        exit(1)
    except Exception as e:
        print_message(f"Error reading queries file {args.queries}: {e}. Exiting.")
        exit(1)

    
    export_colbert_similarity_matrices(
        args.index, 
        args.output_dir, 
        queries_input, 
        args.save_heatmaps
    )




# python ./thesis/similarity_matrices_proce/export_qd_matrixes.py \
#   --index /home/st1084516/ColBERT-Thesis/experiments/fiqa_colbert_tuned4/indexes/fiqa_colbert_tuned4 \
#   --output_dir ./qd_matrices_fiqa_untuned \
#   --queries /home/st1084516/ColBERT-Thesis/fiqa_colbert_format/Queries.tsv \
#   --save_heatmaps

# python ./thesis/similarity_matrices_proce/export_qd_matrixes.py \
#   --index /home/st1084516/ColBERT-Thesis/experiments/CF6/indexes/CF6 \
#   --output_dir ./qd_matrices_cf_tune \
#   --queries /home/st1084516/ColBERT-Thesis/CF_DataSet/Queries.tsv \
#   --save_heatmaps

# python ./thesis/similarity_matrices_proce/export_qd_matrixes.py \
#   --index /home/st1084516/ColBERT-Thesis/experiments/CF7/indexes/CF7 \
#   --output_dir ./qd_matrices_cf_untuned \
#   --queries /home/st1084516/ColBERT-Thesis/CF_DataSet/Queries.tsv \
#   --save_heatmaps

# python ./thesis/similarity_matrices_proce/export_qd_matrixes.py \
#   --index /home/st1084516/ColBERT-Thesis/experiments/CF8/indexes/CF8 \
#   --output_dir ./qd_matrices_cf_Mtuned \
#   --queries /home/st1084516/ColBERT-Thesis/CF_DataSet/Queries.tsv \
#   --save_heatmaps



# python ./thesis/similarity_matrices_proce/export_qd_matrixes.py \
#   --index /home/st1084516/ColBERT-Thesis/experiments/CF11/indexes/CF11 \
#   --output_dir ./qd_matrices_cf_Mediumtuned \
#   --queries /home/st1084516/ColBERT-Thesis/CF_DataSet/Queries.tsv \
#   --save_heatmaps