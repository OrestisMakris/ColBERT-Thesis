import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Modified CNNSimilarity with Enhanced Interaction ---
class CNNSimilarity(nn.Module):
    def __init__(self, embedding_dim, cnn_out_dim=128, fc_hidden_dim=128):
        """
        Args:
            embedding_dim (int): Dimension of input token embeddings.
            cnn_out_dim (int): Output dimension of the CNN layers before pooling.
            fc_hidden_dim (int): Hidden dimension for the final FC layers.
        """
        super(CNNSimilarity, self).__init__()
        # Shared convolutional layers.
        # Input: [B, embed_dim, SeqLen]
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=cnn_out_dim, kernel_size=3, padding=1) # Output cnn_out_dim
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        # self.conv3 = nn.Conv1d(in_channels=256, out_channels=cnn_out_dim, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm1d(cnn_out_dim)
        # self.conv4 = nn.Conv1d(in_channels=cnn_out_dim, out_channels=cnn_out_dim, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm1d(cnn_out_dim)
        # self.conv5 = nn.Conv1d(in_channels=cnn_out_dim, out_channels=cnn_out_dim, kernel_size=3, padding=1)
        
        # Removed conv3 for simplicity, can be added back if needed

        # Fully connected layers for the final similarity score.
        # After pooling, Q and D are [B, cnn_out_dim].
        # We concatenate Q, D, Q*D, |Q-D|. Input size = cnn_out_dim * 4
        fc_input_dim = cnn_out_dim * 4
        self.fc1 = nn.Linear(fc_input_dim, fc_hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(fc_hidden_dim // 2, 1) # Output a single score

    def process_sequence(self, x):
        """ Shared CNN processing + Global Max Pooling """
        # x: [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1) # -> [batch, embed_dim, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Global max pooling over the sequence dimension.
        x = torch.max(x, dim=2)[0] # -> [batch, cnn_out_dim]
        return x

    def forward(self, query, document):
        """
        Args:
            query: Query sequence embeddings [batch, q_len, embed_dim]
            document: Document sequence embeddings [batch, d_len, embed_dim]
        Returns:
            score: Similarity score [batch, 1]
        """
        # Process query and document through shared CNN layers + pooling
        query_pooled = self.process_sequence(query)     # [batch, cnnn_out_dim]
        document_pooled = self.process_sequence(document) # [batch, cnn_ouyt_diim]

        # Enhanced interaction faatures
        element_wise_prod = query_pooled * document_pooled
        element_wise_diff = torch.abs(query_pooled - document_pooled)

        # Concatenate features: [Q, D, Q*D,  |Q-D|]
        combined = torch.cat([
            query_pooled,
            document_pooled,
            element_wise_prod,
            element_wise_diff
        ], dim=1) # -> [batch, cnn_out_dim *  4]

        # Fully connected layers for scoring.
        out = F.relu(self.fc1(combined))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out) # Output raw score (logit) [batch, 1]
        return out


