"""
CNN model for relevance classification of query-document similarity matrices.
Architecture: ldr — two convolutional layers (32→16 channels), AdaptiveMaxPool, Linear(16,1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# class SimpleCNN(nn.Module):
#     """ldr architecture: 2-conv (32→16), BatchNorm, SELU, MaxPool2d, Dropout(0.3), Linear(16,1)."""

#     def __init__(self):
#         super().__init__()
#         self.conv1   = nn.Conv2d(1, 32, kernel_size=6, padding=1)
#         self.bn1     = nn.BatchNorm2d(32)
#         self.conv2   = nn.Conv2d(32, 16, kernel_size=6, padding=1)
#         self.bn2     = nn.BatchNorm2d(16)
#         self.pool    = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1     = nn.Linear(16, 1)

#     def forward(self, x):
#         # x: (B, 1, H, W)
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = self.pool(F.selu(self.bn2(self.conv2(x))))
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)  # (B, 16)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Archived / alternative architectures (kept for reference)
# ---------------------------------------------------------------------------

# arcade — single conv, 32 channels
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 32, 6, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.pool    = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(32, 1)
    def forward(self, x):
        x = self.pool(F.selu(self.bn1(self.conv1(x))))
        x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x).squeeze(-1)

# usurum — single conv, 16 channels
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1   = nn.Conv2d(1, 16, 6, padding=1)
#         self.bn1     = nn.BatchNorm2d(16)
#         self.pool    = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1     = nn.Linear(16, 1)
#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)

# triliza — single conv, 8 channels
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1   = nn.Conv2d(1, 8, 6, padding=1)
#         self.bn1     = nn.BatchNorm2d(8)
#         self.pool    = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.2)
#         self.fc1     = nn.Linear(8, 1)
#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)
