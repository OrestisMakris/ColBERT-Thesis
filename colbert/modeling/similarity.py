import torch
import torch.nn as nn

class CNNSimilarity(nn.Module):
    def __init__(self, embedding_dim):
        super(CNNSimilarity, self).__init__()
        # Two 1D convolution layers to process the embeddings
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layers to produce a final similarity score
        self.fc1 = nn.Linear(64 * 2, 32)  # concatenation of query and document features
        self.fc2 = nn.Linear(32, 1)

    def forward(self, query, document):
        """
        Both query and document are expected to have shape [batch, seq_len, embed_dim]
        """
        # Permute to [batch, embed_dim, seq_len] for Conv1d (which expects channels first)
        query = query.permute(0, 2, 1)
        document = document.permute(0, 2, 1)


        query = torch.relu(self.conv1(query))
        query = torch.relu(self.conv2(query))

        query = torch.max(query, dim=2)[0]

        document = torch.relu(self.conv1(document))
        document = torch.relu(self.conv2(document))
        document = torch.max(document, dim=2)[0]

        combined = torch.cat([query, document], dim=1)

        out = torch.relu(self.fc1(combined))
        # Using sigmoid to bound the similarity score between 0 and 1
        out = torch.sigmoid(self.fc2(out))

        return out