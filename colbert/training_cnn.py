import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn_triplet_dataset import CNNTripletDataset
from modeling.similarity import CNNSimilarity
from tqdm import tqdm  # <-- added tqdm import

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
MARGIN = 0.9  # margin for the ranking loss

def load_embeddings(query_path, doc_path):
    """
    Loads pre-saved embeddings tensors. It is assumed that these tensors are saved in a consistent order.
    """
    query_embeddings = torch.load(query_path)  # e.g., shape [num_queries, seq_len, embed_dim]
    doc_embeddings = torch.load(doc_path)      # e.g., shape [num_documents, seq_len, embed_dim]
    return query_embeddings, doc_embeddings

def main():
    # Check if CUDA is available and print GPU details.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    # Load saved embeddings (update the paths accordingly)
    query_embeddings, doc_embeddings = load_embeddings("../colbert_run/all_query_embeddings.pt", "../colbert_run/all_document_embeddings.pt")
    print(f"Loaded query embeddings: {query_embeddings.shape}")
    print(f"Loaded document embeddings: {doc_embeddings.shape}")

    # Create ID-to-index mappings.
    id2index_q = {i: i for i in range(query_embeddings.size(0))}
    id2index_d = {i: i for i in range(doc_embeddings.size(0))}

    # Create the dataset from a JSONL triples file.
    dataset = CNNTripletDataset("../CF_DataSet/triplets.jsonl", query_embeddings, doc_embeddings, id2index_q, id2index_d)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            pin_memory=True if torch.cuda.is_available() else False,  num_workers=4)

    # Initialize the CNN similarity model.
    embed_dim = query_embeddings.size(-1)
    model = CNNSimilarity(embedding_dim=embed_dim)
    print(f"Using device: {device}")
    model = model.to(device)

    # Define a margin ranking loss and optimizer.
    ranking_loss = nn.MarginRankingLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        # Wrap the dataloader with tqdm for a progress bar.
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            query, pos_doc, neg_doc = batch
            query = query.to(device).float()
            pos_doc = pos_doc.to(device).float()
            neg_doc = neg_doc.to(device).float()

            optimizer.zero_grad()
            sim_pos = model(query, pos_doc).squeeze()  # Expected shape: [B]
            sim_neg = model(query, neg_doc).squeeze()    # Expected shape: [B]

            target = torch.ones(sim_pos.size()).to(device)
            loss = ranking_loss(sim_pos, sim_neg, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "cnn_similarity_model.pt")
    print("CNN similarity network training complete and saved as cnn_similarity_model.pt")

if __name__ == "__main__":
    main()