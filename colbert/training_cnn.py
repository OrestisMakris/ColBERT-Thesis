import torch
import torch.nn as nn
import torch.optim as optim
# Add learning rate scheduler import
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, random_split
from cnn_triplet_dataset import CNNTripletDataset # Re-use dataset
# Make sure you are importing the correct CNNSimilarity class
from old_approach.similarity_old import CNNSimilarity
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