import json
import torch
from torch.utils.data import Dataset

class CNNTripletDataset(Dataset):
    def __init__(self, triplets_file, query_embeddings, doc_embeddings, id2index_q, id2index_d):
        """
        triplets_file: path to a JSONL file where each line is a JSON list: [qid, pid_positive, pid_negative]
        query_embeddings: a torch.Tensor of shape [num_queries, seq_len, embed_dim]
        doc_embeddings:   a torch.Tensor of shape [num_documents, seq_len, embed_dim]
        id2index_q: dict mapping query IDs (from triplets) to index in the query_embeddings tensor
        id2index_d: dict mapping document IDs (from triplets) to index in the doc_embeddings tensor
        """
        self.triplets = []
        with open(triplets_file, "r") as f:
            for line in f:
                # Each line should be a JSON array: [qid, pid_positive, pid_negative]
                self.triplets.append(json.loads(line))
        self.query_embeddings = query_embeddings
        self.doc_embeddings = doc_embeddings
        self.id2index_q = id2index_q
        self.id2index_d = id2index_d

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        qid, pos_pid, neg_pid = self.triplets[idx]

        # Get the corresponding indices from the mapping dictionaries
        q_idx = self.id2index_q[qid]
        pos_idx = self.id2index_d[pos_pid]
        neg_idx = self.id2index_d[neg_pid]

        # Retrieve the embeddings and detach them to remove gradient tracking.
        query = self.query_embeddings[q_idx].detach().clone()    # shape: [seq_len, embed_dim]
        pos_doc = self.doc_embeddings[pos_idx].detach().clone()    # shape: [seq_len, embed_dim]
        neg_doc = self.doc_embeddings[neg_idx].detach().clone()    # shape: [seq_len, embed_dim]

        return query, pos_doc, neg_doc