# import torch
# import torch.nn as nn

# class CNNSimilarity(nn.Module):
#     def __init__(self, embedding_dim):
#         super(CNNSimilarity, self).__init__()
#         # Increased convolutional capacity.
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm1d(128)
        
#         # Fully connected layers for the final similarity score.
#         # After global pooling, query and document features are both [batch, 128].
#         # Their concatenation yields a feature vector of size 256.
#         self.fc1 = nn.Linear(128 * 2, 128)
#         self.dropout1 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(128, 64)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)

#     def forward(self, query, document):
#         """
#         Both query and document are expected to have shape [batch, seq_len, embed_dim].
#         """
#         # Permute to [batch, embed_dim, seq_len] for Conv1d.
#         query = query.permute(0, 2, 1)
#         document = document.permute(0, 2, 1)

#         # Process query.
#         query = torch.relu(self.bn1(self.conv1(query)))
#         query = torch.relu(self.bn2(self.conv2(query)))
#         query = torch.relu(self.bn3(self.conv3(query)))
#         query = torch.relu(self.bn4(self.conv4(query)))
#         # Global max pooling over the sequence dimension.
#         query = torch.max(query, dim=2)[0]

#         # Process document.
#         document = torch.relu(self.bn1(self.conv1(document)))
#         document = torch.relu(self.bn2(self.conv2(document)))
#         document = torch.relu(self.bn3(self.conv3(document)))
#         document = torch.relu(self.bn4(self.conv4(document)))
#         document = torch.max(document, dim=2)[0]

#         # Concatenate features.
#         combined = torch.cat([query, document], dim=1)

#         # Fully connected layers.
#         out = torch.relu(self.fc1(combined))
#         out = self.dropout1(out)
#         out = torch.relu(self.fc2(out))
#         out = self.dropout2(out)
#         out = torch.relu(self.fc3(out))
#         # Using sigmoid to bound the similarity score between 0 and 1.
#         out = torch.sigmoid(self.fc4(out))
#         return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNNSimilarityTriplet(nn.Module):
#     def __init__(self, embedding_dim):
#         super(CNNSimilarityTriplet, self).__init__()
#         # Convolutional encoder.
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, padding=1)
#         # Correct the number of features for bn1 to match conv1's output channels
#         self.bn1 = nn.BatchNorm1d(512) 
#         self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm1d(128)
        
#         # Fully connected head that produces the final embedding.
#         # Global max pooling produces a [batch, 128] vector.
#         self.fc = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 32)
#         )
        
#     def encode(self, x):
#         # x: [batch, seq_len, embed_dim]
#         # Permute for Conv1d: [batch, embed_dim, seq_len]
#         x = x.permute(0, 2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         # Global max pooling over the sequence dimension.
#         x = torch.max(x, dim=2)[0]  # Shape: [batch, 128]
#         x = self.fc(x)             # Shape: [batch, 32]
#         # L2 normalize the embedding.
#         x = F.normalize(x, p=2, dim=1)
#         return x
    
#     def forward(self, x):
#         return self.encode(x)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # --- CNN Token Encoder Class ---
# class CNNTokenEncoder(nn.Module):
#     def __init__(self, embedding_dim, output_dim=128):
#         """
#         Args:
#             embedding_dim: Dimension of the input token embeddings (e.g., 128 from ColBERT).
#             output_dim: Dimension of the output token embeddings after CNN processing.
#         """
#         super(CNNTokenEncoder, self).__init__()
#         # Using a simpler 4-layer CNN for demonstration. Adjust as needed.
#         # Layer dimensions: input_dim -> 256 -> 256 -> 128 -> output_dim
#         self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.conv4 = nn.Conv1d(in_channels=128, out_channels=output_dim, kernel_size=3, padding=1)
#         # No BatchNorm after the last conv layer, similar to some practices. Add if needed.

#     def forward(self, x):
#         """
#         Input x: [batch, seq_len, embed_dim]
#         Output: [batch, seq_len, output_dim] (L2 Normalized)
#         """
#         # Permute for Conv1d: [batch, embed_dim, seq_len]
#         x = x.permute(0, 2, 1)

#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.conv4(x) # No activation after the last layer, or add ReLU if preferred

#         # Permute back to [batch, seq_len, output_dim]
#         x = x.permute(0, 2, 1)
#         # L2 normalize along the feature dimension (last dimension)
#         x = F.normalize(x, p=2, dim=-1)
#         return x

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
        query_pooled = self.process_sequence(query)     # [batch, cnn_out_dim]
        document_pooled = self.process_sequence(document) # [batch, cnn_out_dim]

        # Enhanced interaction features
        element_wise_prod = query_pooled * document_pooled
        element_wise_diff = torch.abs(query_pooled - document_pooled)

        # Concatenate features: [Q, D, Q*D, |Q-D|]
        combined = torch.cat([
            query_pooled,
            document_pooled,
            element_wise_prod,
            element_wise_diff
        ], dim=1) # -> [batch, cnn_out_dim * 4]

        # Fully connected layers for scoring.
        out = F.relu(self.fc1(combined))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out) # Output raw score (logit) [batch, 1]
        return out

# --- Keep other classes if needed ---
# class CNNSimilarityTriplet(nn.Module): ...
# class CNNTokenEncoder(nn.Module): ...