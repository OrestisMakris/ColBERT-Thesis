# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from cnn_triplet_dataset import CNNTripletDataset
# from modeling.similarity import CNNSimilarity
# from tqdm import tqdm  # <-- added tqdm import

# # Hyperparameters
# BATCH_SIZE = 16
# NUM_EPOCHS = 80
# LEARNING_RATE = 1e-3
# # Using SoftMarginLoss, so no explicit margin is needed.

# def load_embeddings(query_path, doc_path):
#     """
#     Loads pre-saved embeddings tensors.
#     query_embeddings: shape [num_queries, seq_len, embed_dim]
#     doc_embeddings:   shape [num_documents, seq_len, embed_dim]
#     """
#     query_embeddings = torch.load(query_path)  # e.g., shape [num_queries, seq_len, embed_dim]
#     doc_embeddings = torch.load(doc_path)      # e.g., shape [num_documents, seq_len, embed_dim]
    
#     # Ensure embeddings are on CPU so that DataLoader workers can use them without GPU initialization issues.
#     query_embeddings = query_embeddings.cpu()
#     doc_embeddings = doc_embeddings.cpu()
    
#     return query_embeddings, doc_embeddings

# def main():
#     # Check if CUDA is available and print GPU details.
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")
    
#     # Load query and document embeddings from the exported files.
#     query_embeddings, doc_embeddings = load_embeddings("../colbert_run/exported_all_query.pt", 
#                                                          "../colbert_run/exported_all_doc_padded.pt")
#     print(f"Loaded query embeddings: {query_embeddings.shape}")
#     print(f"Loaded document embeddings: {doc_embeddings.shape}")

#     # Create ID-to-index mappings.
#     id2index_q = {i: i for i in range(query_embeddings.size(0))}
#     id2index_d = {i: i for i in range(doc_embeddings.size(0))}

#     # Create the dataset using a JSONL triples file.
#     dataset = CNNTripletDataset("../CF_DataSet/triplets.jsonl", query_embeddings, doc_embeddings, id2index_q, id2index_d)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
#                               pin_memory=True if torch.cuda.is_available() else False, num_workers=4)

#     # Initialize the CNN similarity model.
#     embed_dim = query_embeddings.size(-1)
#     model = CNNSimilarity(embedding_dim=embed_dim)
#     print(f"Using device: {device}")
#     model = model.to(device)

#     # Use SoftMarginLoss as an alternative ranking loss.
#     # The idea here: We want sim_pos > sim_neg. So we compute (sim_pos - sim_neg)
#     # and use SoftMarginLoss with target labels 1.
#     ranking_loss = nn.SoftMarginLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     model.train()
#     for epoch in range(NUM_EPOCHS):
#         running_loss = 0.0
#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
#             query, pos_doc, neg_doc = batch
#             query = query.to(device).float()
#             pos_doc = pos_doc.to(device).float()
#             neg_doc = neg_doc.to(device).float()

#             optimizer.zero_grad()
#             sim_pos = model(query, pos_doc).squeeze()  # Expected shape: [B]
#             sim_neg = model(query, neg_doc).squeeze()    # Expected shape: [B]
            
#             # Compute the score difference and assign a positive label (1).
#             score_diff = sim_pos - sim_neg
#             labels = torch.ones_like(score_diff).to(device)
#             loss = ranking_loss(score_diff, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

#     torch.save(model.state_dict(), "cnn_similarity_model.pt")
#     print("CNN similarity network training complete and saved as cnn_similarity_model.pt")

# if __name__ == "__main__":
#     main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from cnn_triplet_dataset import CNNTripletDataset
# from modeling.similarity import CNNSimilarityTriplet
# from tqdm import tqdm

# # Hyperparameters
# BATCH_SIZE = 8
# NUM_EPOCHS = 40
# LEARNING_RATE = 1e-2
# MARGIN = 1.0  # margin for triplet loss

# def load_embeddings(query_path, doc_path):
#     """
#     Loads pre-saved embeddings tensors.
#     query_embeddings: shape [num_queries, seq_len, embed_dim]
#     doc_embeddings:   shape [num_documents, seq_len, embed_dim]
#     """
#     query_embeddings = torch.load(query_path).cpu()
#     doc_embeddings = torch.load(doc_path).cpu()    
#     return query_embeddings, doc_embeddings

# def main():
#     # Device configuration.
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")
    
#     # Load embeddings.
#     query_embeddings, doc_embeddings = load_embeddings("../colbert_run/exported_all_query.pt", 
#                                                          "../colbert_run/exported_all_doc_padded.pt")
#     print(f"Loaded query embeddings: {query_embeddings.shape}")
#     print(f"Loaded document embeddings: {doc_embeddings.shape}")
    
#     # Create ID-to-index mappings.
#     id2index_q = {i: i for i in range(query_embeddings.size(0))}
#     id2index_d = {i: i for i in range(doc_embeddings.size(0))}
    
#     # Create the dataset using a JSONL file with triplets.
#     dataset = CNNTripletDataset("../CF_DataSet/triplets.jsonl", query_embeddings, doc_embeddings, id2index_q, id2index_d)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
#                               pin_memory=True if torch.cuda.is_available() else False, num_workers=4)
    
#     # Initialize the triplet model.
#     embed_dim = query_embeddings.size(-1)
#     model = CNNSimilarityTriplet(embedding_dim=embed_dim)
#     model = model.to(device)
#     print(f"Model initialized on device: {device}")
    
#     # Define optimizer and TripletMarginLoss.
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     triplet_loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
    
#     model.train()
#     for epoch in range(NUM_EPOCHS):
#         running_loss = 0.0
#         for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
#             query, pos_doc, neg_doc = batch
#             query = query.to(device).float()
#             pos_doc = pos_doc.to(device).float()
#             neg_doc = neg_doc.to(device).float()
            
#             # Compute embeddings.
#             anchor_embed = model(query)
#             positive_embed = model(pos_doc)
#             negative_embed = model(neg_doc)
            
#             # Compute TripletMarginLoss.
#             loss = triplet_loss_fn(anchor_embed, positive_embed, negative_embed)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        
#         avg_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    
#     torch.save(model.state_dict(), "cnn_similarity_triplet_model.pt")
#     print("Triplet network training complete and saved as cnn_similarity_triplet_model.pt")

# if __name__ == "__main__":
#     main()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset, random_split
# from cnn_triplet_dataset import CNNTripletDataset # Re-use dataset, it provides triplets
# from modeling.similarity import CNNTokenEncoder # Import only the encoder
# from tqdm import tqdm
# import os

# # --- Hyperparameters ---
# BATCH_SIZE = 16 # Adjust based on GPU memory, MaxSim can be memory intensive
# NUM_EPOCHS = 10 # Adjust as needed
# LEARNING_RATE = 5e-5 # May need tuning
# CNN_OUTPUT_DIM = 128 # Output dimension of the CNN token embeddings
# VALIDATION_SPLIT = 0.1 # Use 10% of data for validation
# MODEL_SAVE_PATH = "cnn_token_encoder.pt"
# BEST_MODEL_SAVE_PATH = "cnn_token_encoder_best.pt"

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
#     sim_matrix = torch.bmm(Q_tokens, D_tokens.permute(0, 2, 1)) # [batch, q_len, d_len]
#     max_sim, _ = torch.max(sim_matrix, dim=2) # [batch, q_len]
#     scores = torch.sum(max_sim, dim=1) # [batch]
#     return scores

# # --- Main Training Function ---
# def main():
#     # Device configuration
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
#     else:
#         device = torch.device("cpu")
#         print("CUDA is not available. Using CPU.")

#     # Load original embeddings
#     print("Loading original embeddings...")
#     query_embeddings_orig = load_embeddings("../colbert_run/exported_all_query.pt")
#     doc_embeddings_orig = load_embeddings("../colbert_run/exported_all_doc_padded.pt")
#     print(f"Loaded query embeddings: {query_embeddings_orig.shape}")
#     print(f"Loaded document embeddings: {doc_embeddings_orig.shape}")

#     # Create ID-to-index mappings
#     id2index_q = {i: i for i in range(query_embeddings_orig.size(0))}
#     id2index_d = {i: i for i in range(doc_embeddings_orig.size(0))}

#     # Create the full dataset
#     full_dataset = CNNTripletDataset("../CF_DataSet/triplets.jsonl", query_embeddings_orig, doc_embeddings_orig, id2index_q, id2index_d)

#     # Split dataset into training and validation
#     num_samples = len(full_dataset)
#     val_size = int(VALIDATION_SPLIT * num_samples)
#     train_size = num_samples - val_size
#     print(f"Splitting data: {train_size} train, {val_size} validation")
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#     train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
#                                   pin_memory=True if torch.cuda.is_available() else False, num_workers=4)
#     val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
#                                 pin_memory=True if torch.cuda.is_available() else False, num_workers=4)

#     # Initialize the CNN Token Encoder
#     input_embed_dim = query_embeddings_orig.size(-1)
#     cnn_encoder = CNNTokenEncoder(embedding_dim=input_embed_dim, output_dim=CNN_OUTPUT_DIM).to(device)

#     # Define optimizer and Pairwise Ranking Loss (SoftMarginLoss)
#     optimizer = optim.Adam(cnn_encoder.parameters(), lr=LEARNING_RATE)
#     ranking_loss_fn = nn.SoftMarginLoss()

#     best_val_loss = float('inf')

#     # --- Training Loop ---
#     print("Starting training...")
#     for epoch in range(NUM_EPOCHS):
#         # --- Training Phase ---
#         cnn_encoder.train()
#         running_train_loss = 0.0
#         for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
#             query, pos_doc, neg_doc = batch
#             query = query.to(device).float()
#             pos_doc = pos_doc.to(device).float()
#             neg_doc = neg_doc.to(device).float()

#             optimizer.zero_grad()

#             # Encode tokens using the CNN
#             query_tokens_cnn = cnn_encoder(query)     # [B, q_len, cnn_dim]
#             pos_doc_tokens_cnn = cnn_encoder(pos_doc) # [B, d_len, cnn_dim]
#             neg_doc_tokens_cnn = cnn_encoder(neg_doc) # [B, d_len, cnn_dim]

#             # Calculate MaxSim scores directly
#             score_pos = calculate_maxsim_score(query_tokens_cnn, pos_doc_tokens_cnn) # [B]
#             score_neg = calculate_maxsim_score(query_tokens_cnn, neg_doc_tokens_cnn) # [B]

#             # Compute pairwise ranking loss
#             score_diff = score_pos - score_neg
#             labels = torch.ones_like(score_diff).to(device) # Target is 1 for SoftMarginLoss
#             loss = ranking_loss_fn(score_diff, labels)

#             loss.backward()
#             optimizer.step()

#             running_train_loss += loss.item()

#         avg_train_loss = running_train_loss / len(train_dataloader)

#         # --- Validation Phase ---
#         cnn_encoder.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
#                 query, pos_doc, neg_doc = batch
#                 query = query.to(device).float()
#                 pos_doc = pos_doc.to(device).float()
#                 neg_doc = neg_doc.to(device).float()

#                 # Encode tokens
#                 query_tokens_cnn = cnn_encoder(query)
#                 pos_doc_tokens_cnn = cnn_encoder(pos_doc)
#                 neg_doc_tokens_cnn = cnn_encoder(neg_doc)

#                 # Calculate scores directly
#                 score_pos = calculate_maxsim_score(query_tokens_cnn, pos_doc_tokens_cnn)
#                 score_neg = calculate_maxsim_score(query_tokens_cnn, neg_doc_tokens_cnn)

#                 # Compute loss
#                 score_diff = score_pos - score_neg
#                 labels = torch.ones_like(score_diff).to(device)
#                 loss = ranking_loss_fn(score_diff, labels)
#                 running_val_loss += loss.item()

#         avg_val_loss = running_val_loss / len(val_dataloader)
#         print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#         # Save the best model based on validation loss
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(cnn_encoder.state_dict(), BEST_MODEL_SAVE_PATH)
#             print(f"   > New best model saved to {BEST_MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

#         # Optionally save the model from the last epoch
#         torch.save(cnn_encoder.state_dict(), MODEL_SAVE_PATH)

#     print(f"Training complete. Last model saved to {MODEL_SAVE_PATH}, Best model saved to {BEST_MODEL_SAVE_PATH}")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import torch.optim as optim
# Add learning rate scheduler import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
from cnn_triplet_dataset import CNNTripletDataset # Re-use dataset
# Make sure you are importing the correct CNNSimilarity class
from modeling.similarity import CNNSimilarity
from tqdm import tqdm
import os

# --- Hyperparameters ---
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4 # Keep reduced LR
WEIGHT_DECAY = 1e-5 # Keep weight decay
MARGIN = 0.5 # <<< Define Margin for MarginRankingLoss
VALIDATION_SPLIT = 0.1
MODEL_SAVE_PATH = "cnn_pairwise_scorer_leaky_margin.pt" # New save name
BEST_MODEL_SAVE_PATH = "cnn_pairwise_scorer_leaky_margin_best.pt" # New save name
# Scheduler params
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5

# --- Utility Functions ---
# ... (load_embeddings function remains the same) ...
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

# --- Main Training Function ---
def main():
    # Device configuration
    # ... (device setup remains the same) ...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Load original embeddings
    # ... (loading embeddings remains the same) ...
    print("Loading original embeddings...")
    query_embeddings_orig = load_embeddings("../colbert_run/exported_all_query.pt")
    doc_embeddings_orig = load_embeddings("../colbert_run/exported_all_doc_padded.pt")
    print(f"Loaded query embeddings: {query_embeddings_orig.shape}")
    print(f"Loaded document embeddings: {doc_embeddings_orig.shape}")

    # Create ID-to-index mappings
    # ... (id2index remains the same) ...
    id2index_q = {i: i for i in range(query_embeddings_orig.size(0))}
    id2index_d = {i: i for i in range(doc_embeddings_orig.size(0))}

    # Create the full dataset
    # ... (dataset creation remains the same) ...
    full_dataset = CNNTripletDataset("../CF_DataSet/triplets.jsonl", query_embeddings_orig, doc_embeddings_orig, id2index_q, id2index_d)

    # Split dataset into training and validation
    # ... (dataset split remains the same) ...
    num_samples = len(full_dataset)
    val_size = int(VALIDATION_SPLIT * num_samples)
    train_size = num_samples - val_size
    print(f"Splitting data: {train_size} train, {val_size} validation")
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True if torch.cuda.is_available() else False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                pin_memory=True if torch.cuda.is_available() else False, num_workers=4)

    # Initialize the CNN Pairwise Scorer model
    input_embed_dim = query_embeddings_orig.size(-1)
    # Instantiate the modified CNNSimilarity
    model = CNNSimilarity(embedding_dim=input_embed_dim).to(device)

    # Define optimizer (with new LR and weight decay) and Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # <<< CHANGE LOSS FUNCTION >>>
    ranking_loss_fn = nn.MarginRankingLoss(margin=MARGIN)

    # Define Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)

    best_val_loss = float('inf')

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            query, pos_doc, neg_doc = batch
            query = query.to(device).float()
            pos_doc = pos_doc.to(device).float()
            neg_doc = neg_doc.to(device).float()

            optimizer.zero_grad()

            score_pos = model(query, pos_doc).squeeze(-1) # [B]
            score_neg = model(query, neg_doc).squeeze(-1) # [B]

            # <<< ADJUST LOSS CALCULATION for MarginRankingLoss >>>
            # Target is 1, indicating score_pos should be greater than score_neg by margin
            target = torch.ones_like(score_pos).to(device)
            loss = ranking_loss_fn(score_pos, score_neg, target)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                query, pos_doc, neg_doc = batch
                query = query.to(device).float()
                pos_doc = pos_doc.to(device).float()
                neg_doc = neg_doc.to(device).float()

                score_pos = model(query, pos_doc).squeeze(-1)
                score_neg = model(query, neg_doc).squeeze(-1)

                # <<< ADJUST LOSS CALCULATION for MarginRankingLoss >>>
                target = torch.ones_like(score_pos).to(device)
                loss = ranking_loss_fn(score_pos, score_neg, target)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Scheduler Step ---
        scheduler.step(avg_val_loss)

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            print(f"   > New best model saved to {BEST_MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.4f})")

        # Optionally save the model from the last epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training complete. Last model saved to {MODEL_SAVE_PATH}, Best model saved to {BEST_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()