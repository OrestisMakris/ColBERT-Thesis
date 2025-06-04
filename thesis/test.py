import torch
from colbert.modeling.colbert import ColBERT
from colbert.infra.config.config import ColBERTConfig
from torch.utils.data import DataLoader
from colbert.dataset.CNNTripletDataset import CNNTripletDataset  # Adjust if needed

# Set up your configuration and dataset paths
config = ColBERTConfig(/* … appropriate parameters … */)
colbert_model = ColBERT(colbert_config=config)
colbert_model.eval()  # Inference mode

dataset = CNNTripletDataset(triplets_file='path/to/triplets.jsonl',
                            queries='path/to/Queries.tsv',
                            collection='path/to/docs.tsv',
                            config=config)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

all_query_embeddings = []
all_document_embeddings = []

for batch in dataloader:
    # Assume batch provides (q_input, d_input) appropriately.
    q_input, d_input = batch[0], batch[1]
    # Run ColBERT to produce query embeddings and document embeddings.
    Q = colbert_model.query(*q_input)  # e.g., shape [1, query_length, embed_dim]
    D, _ = colbert_model.doc(*d_input, keep_dims='return_mask')  # e.g., shape [2, doc_length, embed_dim]
    
    all_query_embeddings.append(Q.squeeze(0))  
    all_document_embeddings.append(D)  # Adjust squeeze if needed

# Concatenate all computed embeddings
all_query_embeddings_tensor = torch.cat(all_query_embeddings, dim=0)
all_document_embeddings_tensor = torch.cat(all_document_embeddings, dim=0)

torch.save(all_query_embeddings_tensor, 'all_query_embeddings.pt')
torch.save(all_document_embeddings_tensor, 'all_document_embeddings.pt')

print("Saved embeddings.")