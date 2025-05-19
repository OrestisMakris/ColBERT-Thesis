import torch
import torch.nn as nn
from modeling.similarity import CNNSimilarity # Import the scoring model
from tqdm import tqdm
import numpy as np
import os

# --- Utility Functions ---
def load_embeddings(path, weights_only=True):
    """Loads embeddings, recommending weights_only=True for safety."""
    try:
        embeddings = torch.load(path, map_location='cpu', weights_only=weights_only)
        return embeddings
    except Exception as e:
        if weights_only:
            print(f"Warning: Failed loading {path} with weights_only=True. Retrying with weights_only=False. Error: {e}")
            return torch.load(path, map_loQcation='cpu', weights_only=False)
        else:
            raise e

# --- Main Inference Function ---
def main_inference(model_path, query_emb_path, doc_emb_path, output_ranking_path, batch_size=128):
    # --- 1. Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # --- 2. Load Original Embeddings ---
    print("Loading original embeddings...")
    query_embeddings_orig = load_embeddings(query_emb_path)
    doc_embeddings_orig = load_embeddings(doc_emb_path)
    print(f"Loaded original query embeddings: {query_embeddings_orig.shape}")
    print(f"Loaded original document embeddings: {doc_embeddings_orig.shape}")

    num_queries = query_embeddings_orig.size(0)
    num_docs = doc_embeddings_orig.size(0)
    input_embed_dim = query_embeddings_orig.size(-1)

    # --- 3. Load Trained CNN Pairwise Scorer Model ---
    print(f"Loading trained CNN Pairwise Scorer model from {model_path}...")
    model = CNNSimilarity(embedding_dim=input_embed_dim)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Warning: Failed loading model state dict with weights_only=True. Retrying with weights_only=False. Error: {e}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 4. Calculate Scores & Write Rankings (Query by Query) ---
    print(f"Calculating scores and writing rankings to {output_ranking_path}...")
    os.makedirs(os.path.dirname(output_ranking_path), exist_ok=True)

    with open(output_ranking_path, 'w') as f_out, torch.no_grad():
        for q_idx in tqdm(range(num_queries)):
            # Get the single query embedding sequence
            query_orig_emb = query_embeddings_orig[q_idx:q_idx+1].to(device).float() # [1, q_len, dim]

            # Calculate scores against all documents (batching document processing)
            scores_for_q = []
            for i in range(0, num_docs, batch_size):
                # Get batch of document embedding sequences
                doc_batch_orig_emb = doc_embeddings_orig[i:i+batch_size].to(device).float() # [B, d_len, dim]
                current_batch_size = doc_batch_orig_emb.size(0)

                # Expand query to match document batch size
                query_orig_emb_expanded = query_orig_emb.expand(current_batch_size, -1, -1) # [B, q_len, dim]

                # Calculate scores for the batch using the pairwise model
                batch_scores = model(query_orig_emb_expanded, doc_batch_orig_emb).squeeze(-1) # [B]
                scores_for_q.append(batch_scores.cpu()) # Move scores back to CPU

            scores_for_q = torch.cat(scores_for_q, dim=0) # Shape: [num_docs]

            # Sort scores to get rank and original doc indices
            ranked_scores, ranked_indices = torch.sort(scores_for_q, descending=True)

            # Write top N results
            max_rank = 1000 # Standard for MS MARCO eval
            for rank in range(min(max_rank, num_docs)):
                doc_id = ranked_indices[rank].item() # doc_id is the original index
                score = ranked_scores[rank].item()
                # Format: qid \t pid \t rank \t score
                f_out.write(f"{q_idx}\t{doc_id}\t{rank + 1}\t{score:.6f}\n")

    print(f"Inference and ranking complete. Output saved to {output_ranking_path}")

if __name__ == "__main__":
    # --- Configuration ---
    # Use the best model saved during training
    MODEL_PATH = "cnn_pairwise_scorer_leaky_margin.pt"
    QUERY_EMB_PATH = "../colbert_run/exported_all_query.pt"
    DOC_EMB_PATH = "../colbert_run/exported_all_doc_padded.pt"
    # Define the output path for the ranking file
    OUTPUT_RANKING_PATH = "rankings/cnn_pairwise.ranking.tsv"
    # Batch size for inference (adjust based on GPU memory)
    INFERENCE_BATCH_SIZE = 256# Can often be larger here

    # --- Run Inference ---
    main_inference(MODEL_PATH, QUERY_EMB_PATH, DOC_EMB_PATH, OUTPUT_RANKING_PATH,
                   batch_size=INFERENCE_BATCH_SIZE)