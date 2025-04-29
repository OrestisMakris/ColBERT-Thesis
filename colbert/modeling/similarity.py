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

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSimilarityTriplet(nn.Module):
    def __init__(self, embedding_dim):
        super(CNNSimilarityTriplet, self).__init__()
        # Convolutional encoder.
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        # Fully connected head that produces the final embedding.
        # Global max pooling produces a [batch, 128] vector.
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32)
        )
        
    def encode(self, x):
        # x: [batch, seq_len, embed_dim]
        # Permute for Conv1d: [batch, embed_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # Global max pooling over the sequence dimension.
        x = torch.max(x, dim=2)[0]  # Shape: [batch, 128]
        x = self.fc(x)             # Shape: [batch, 32]
        # L2 normalize the embedding.
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x):
        return self.encode(x)