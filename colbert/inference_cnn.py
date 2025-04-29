# import torch
# import torch.nn as nn
# from modeling.similarity import CNNSimilarityTriplet # Import your model class
# from tqdm import tqdm
# import numpy as np
# import os

# def load_embeddings(path, weights_only=True):
#     """Loads embeddings, recommending weights_only=True for safety."""
#     try:
#         embeddings = torch.load(path, map_location='cpu', weights_only=weights_only)
#         return embeddings
#     except Exception as e:
#         if weights_only:
#             print(f"Warning: Failed loading {path} with weights_only=True. Retrying with weights_only=False. Error: {e}")
#             return torch.load(path, map_location='cpu', weights_only=False)
#         else:
#             raise e

# def main_inference(model_path, query_emb_path, doc_emb_path, output_ranking_path, batch_size=128):
#     # -Setup Device ---
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")

#     # -Load Embeddings ---
#     print("Loading original embeddings...")
#     query_embeddings_orig = load_embeddings(query_emb_path)
#     doc_embeddings_orig = load_embeddings(doc_emb_path)
#     print(f"Loaded original query embeddings: {query_embeddings_orig.shape}")
#     print(f"Loaded original document embeddings: {doc_embeddings_orig.shape}")

#     num_queries = query_embeddings_orig.size(0)
#     num_docs = doc_embeddings_orig.size(0)
#     embedding_dim = query_embeddings_orig.size(-1)

#     # ---  Load Trained CNN Model ---
#     print("Loading trained CNN model...")
#     model = CNNSimilarityTriplet(embedding_dim=embedding_dim)
#     try:
#         model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     except Exception as e:
#         print(f"Warning: Failed loading model state dict with weights_only=True. Retrying with weights_only=False. Error: {e}")
#         model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

#     model = model.to(device)
#     model.eval()
#     print("Model loaded successfully.")

#     # Encode Queries and Documents using the CNN ---
#     all_query_cnn_embeddings = []
#     all_doc_cnn_embeddings = []

#     with torch.no_grad():
#         print("Encoding queries with CNN...")
#         for i in tqdm(range(0, num_queries, batch_size)):
#             batch_q = query_embeddings_orig[i:i+batch_size].to(device).float()
#             cnn_embeds_q = model(batch_q)
#             all_query_cnn_embeddings.append(cnn_embeds_q.cpu())

#         print("Encoding documents with CNN...")
#         for i in tqdm(range(0, num_docs, batch_size)):
#             batch_d = doc_embeddings_orig[i:i+batch_size].to(device).float()
#             cnn_embeds_d = model(batch_d)
#             all_doc_cnn_embeddings.append(cnn_embeds_d.cpu())

#     query_cnn_embeddings = torch.cat(all_query_cnn_embeddings, dim=0)
#     doc_cnn_embeddings = torch.cat(all_doc_cnn_embeddings, dim=0)
#     print(f"Encoded queries shape: {query_cnn_embeddings.shape}")
#     print(f"Encoded documents shape: {doc_cnn_embeddings.shape}")

#     # Calculate Similarity Scores & Write Rankings ---
#     print(f"Calculating scores and writing rankings to {output_ranking_path}...")
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(output_ranking_path), exist_ok=True)

#     with open(output_ranking_path, 'w') as f_out:
#         # Process one query at a time to manage memory for large score matrices
#         for q_idx in tqdm(range(num_queries)):
#             # Move current query embedding to GPU
#             query_emb = query_cnn_embeddings[q_idx:q_idx+1].to(device) # Shape [1, 32]

#             # Calculate scores for this query against all docs (potentially in batches)
#             scores_for_q = []
#             for i in range(0, num_docs, batch_size):
#                 doc_batch_emb = doc_cnn_embeddings[i:i+batch_size].to(device) # Shape [batch_size, 32]
#                 batch_scores = torch.matmul(query_emb, doc_batch_emb.t()) # Shape [1, batch_size]
#                 scores_for_q.append(batch_scores.cpu())

#             scores_for_q = torch.cat(scores_for_q, dim=1).squeeze(0) # Shape: [num_docs]

#             # Sort scores to get rank and original doc indices
#             ranked_scores, ranked_indices = torch.sort(scores_for_q, descending=True)

#             # Write top N results (e.g., top 1000 like MS MARCO standard)
#             max_rank = 1000 # Or set to num_docs if you want all
#             for rank in range(min(max_rank, num_docs)):
#                 doc_id = ranked_indices[rank].item()
#                 score = ranked_scores[rank].item()
#                 # Write line: qid \t pid \t rank \t score
#                 f_out.write(f"{q_idx}\t{doc_id}\t{rank + 1}\t{score:.6f}\n")

#     print(f"Inference and ranking complete. Output saved to {output_ranking_path}")

# if __name__ == "__main__":
#     MODEL_PATH = "cnn_similarity_triplet_model.pt"
#     QUERY_EMB_PATH = "../colbert_run/exported_all_query.pt"
#     DOC_EMB_PATH = "../colbert_run/exported_all_doc_padded.pt"
#     # Define the output path for the ranking file
#     OUTPUT_RANKING_PATH = "rankings/cnn_triplet.ranking.tsv"
#     INFERENCE_BATCH_SIZE =1 # Adjust based on GPU memory

#     main_inference(MODEL_PATH, QUERY_EMB_PATH, DOC_EMB_PATH, OUTPUT_RANKING_PATH, batch_size=INFERENCE_BATCH_SIZE)


# import torch
# import torch.nn as nn
# from modeling.similarity import CNNTokenEncoder # Import only the encoder
# from tqdm import tqdm
# import numpy as np
# import os

# # --- Utility Functions ---
# def load_embeddings(path, weights_only=True):
#     """Loads embeddings, recommending weights_only=True for safety."""
#     try:
#         embeddings = torch.load(path, map_location='cpu', weights_only=weights_only)
#         return embeddings
#     except Exception as e:
#         if weights_only:
#             print(f"Warning: Failed loading {path} with weights_only=True. Retrying with weights_only=False. Error: {e}")
#             return torch.load(path, map_location='cpu', weights_only=False)
#         else:
#             raise e

# def calculate_maxsim_score(Q_tokens, D_tokens):
#     """
#     Calculates the ColBERT MaxSim score directly.
#     Assumes input embeddings Q_tokens and D_tokens are L2 normalized.
#     Args:
#         Q_tokens: Query token embeddings [batch, q_len, dim]
#         D_tokens: Document token embeddings [batch, d_len, dim]
#     Returns:
#         scores: MaxSim scores [batch]
#     """
#     # Ensure batch dimensions match for bmm
#     if Q_tokens.size(0) != D_tokens.size(0):
#         if Q_tokens.size(0) == 1:
#             Q_tokens = Q_tokens.expand(D_tokens.size(0), -1, -1)
#         elif D_tokens.size(0) == 1:
#              D_tokens = D_tokens.expand(Q_tokens.size(0), -1, -1)
#         else:
#             raise ValueError(f"Batch dimensions must match or one must be 1. Got Q: {Q_tokens.shape}, D: {D_tokens.shape}")

#     sim_matrix = torch.bmm(Q_tokens, D_tokens.permute(0, 2, 1)) # [batch, q_len, d_len]
#     max_sim, _ = torch.max(sim_matrix, dim=2) # [batch, q_len]
#     scores = torch.sum(max_sim, dim=1) # [batch]
#     return scores

# # --- Main Inference Function ---
# def main_inference(model_path, query_emb_path, doc_emb_path, output_ranking_path, batch_size=128, cnn_output_dim=128):
#     # --- 1. Setup Device ---
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")

#     # --- 2. Load Original Embeddings ---
#     print("Loading original embeddings...")
#     query_embeddings_orig = load_embeddings(query_emb_path)
#     doc_embeddings_orig = load_embeddings(doc_emb_path)
#     print(f"Loaded original query embeddings: {query_embeddings_orig.shape}")
#     print(f"Loaded original document embeddings: {doc_embeddings_orig.shape}")

#     num_queries = query_embeddings_orig.size(0)
#     num_docs = doc_embeddings_orig.size(0)
#     input_embed_dim = query_embeddings_orig.size(-1)

#     # --- 3. Load Trained CNN Token Encoder ---
#     print(f"Loading trained CNN Token Encoder model from {model_path}...")
#     cnn_encoder = CNNTokenEncoder(embedding_dim=input_embed_dim, output_dim=cnn_output_dim)
#     try:
#         # Try loading with weights_only=True first for security
#         cnn_encoder.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     except Exception as e:
#         print(f"Warning: Failed loading model state dict with weights_only=True. Retrying with weights_only=False. Error: {e}")
#         # Fallback to weights_only=False if the first attempt fails
#         cnn_encoder.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

#     cnn_encoder = cnn_encoder.to(device)
#     cnn_encoder.eval()
#     print("Model loaded successfully.")

#     # --- 4. Pre-encode All Documents using the CNN ---
#     # Store encoded documents on CPU to conserve GPU memory during query processing
#     all_doc_cnn_tokens_cpu = []
#     with torch.no_grad():
#         print("Encoding all documents with CNN...")
#         for i in tqdm(range(0, num_docs, batch_size)):
#             batch_d = doc_embeddings_orig[i:i+batch_size].to(device).float()
#             cnn_tokens_d = cnn_encoder(batch_d) # [B, d_len, cnn_dim]
#             all_doc_cnn_tokens_cpu.append(cnn_tokens_d.cpu()) # Store on CPU

#     # --- 5. Calculate Scores & Write Rankings (Query by Query) ---
#     print(f"Calculating scores and writing rankings to {output_ranking_path}...")
#     os.makedirs(os.path.dirname(output_ranking_path), exist_ok=True)

#     with open(output_ranking_path, 'w') as f_out, torch.no_grad():
#         for q_idx in tqdm(range(num_queries)):
#             # Encode the current query
#             query_orig_emb = query_embeddings_orig[q_idx:q_idx+1].to(device).float()
#             query_cnn_tokens = cnn_encoder(query_orig_emb) # [1, q_len, cnn_dim]

#             # Calculate scores against all documents (batching document processing)
#             scores_for_q = []
#             for doc_batch_tokens_cpu in all_doc_cnn_tokens_cpu:
#                 # Move the current batch of document tokens to GPU
#                 doc_batch_tokens = doc_batch_tokens_cpu.to(device)

#                 # Calculate scores for the batch using direct MaxSim function
#                 # query_cnn_tokens has batch size 1, doc_batch_tokens has batch size B
#                 batch_scores = calculate_maxsim_score(query_cnn_tokens, doc_batch_tokens) # [B]
#                 scores_for_q.append(batch_scores.cpu()) # Move scores back to CPU

#             scores_for_q = torch.cat(scores_for_q, dim=0) # Shape: [num_docs]

#             # Sort scores to get rank and original doc indices
#             ranked_scores, ranked_indices = torch.sort(scores_for_q, descending=True)

#             # Write top N results
#             max_rank = 1000 # Standard for MS MARCO eval
#             for rank in range(min(max_rank, num_docs)):
#                 doc_id = ranked_indices[rank].item() # doc_id is the original index from doc_embeddings_orig
#                 score = ranked_scores[rank].item()
#                 # Format: qid \t pid \t rank \t score
#                 f_out.write(f"{q_idx}\t{doc_id}\t{rank + 1}\t{score:.6f}\n")

#     print(f"Inference and ranking complete. Output saved to {output_ranking_path}")

# if __name__ == "__main__":
#     # --- Configuration ---
#     # Use the best model saved during training (make sure path is correct)
#     MODEL_PATH = "cnn_token_encoder_best.pt"
#     QUERY_EMB_PATH = "../colbert_run/exported_all_query.pt"
#     DOC_EMB_PATH = "../colbert_run/exported_all_doc_padded.pt"
#     # Define the output path for the ranking file
#     OUTPUT_RANKING_PATH = "rankings/cnn_token.ranking.tsv"
#     # Batch size for encoding/inference (adjust based on GPU memory)
#     INFERENCE_BATCH_SIZE = 256
#     # Must match the output dim used during training
#     # Check your training script (`training_cnn_token.py`) for CNN_OUTPUT_DIM
#     CNN_OUTPUT_DIM_INFERENCE = 128

#     # --- Run Inference ---
#     main_inference(MODEL_PATH, QUERY_EMB_PATH, DOC_EMB_PATH, OUTPUT_RANKING_PATH,
#                    batch_size=INFERENCE_BATCH_SIZE, cnn_output_dim=CNN_OUTPUT_DIM_INFERENCE)

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
            return torch.load(path, map_location='cpu', weights_only=False)
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
    MODEL_PATH = "cnn_pairwise_scorer_best.pt"
    QUERY_EMB_PATH = "../colbert_run/exported_all_query.pt"
    DOC_EMB_PATH = "../colbert_run/exported_all_doc_padded.pt"
    # Define the output path for the ranking file
    OUTPUT_RANKING_PATH = "rankings/cnn_pairwise.ranking.tsv"
    # Batch size for inference (adjust based on GPU memory)
    INFERENCE_BATCH_SIZE = 256 # Can often be larger here

    # --- Run Inference ---
    main_inference(MODEL_PATH, QUERY_EMB_PATH, DOC_EMB_PATH, OUTPUT_RANKING_PATH,
                   batch_size=INFERENCE_BATCH_SIZE)