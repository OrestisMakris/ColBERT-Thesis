import torch
import torch.nn as nn
import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Assuming you want 5x5 kernels as per previous request
#         self.conv1 = nn.Conv2d(1, 32, 5, padding=1)  # 1 input channel, 32 output channels, 5x5 kernel
#         self.bn1   = nn.BatchNorm2d(32)
#         # self.conv2 = nn.Conv2d(4, 8, 5, padding=1) #5 map 26 mrr 75
#         # self.bn2   = nn.BatchNorm2d(8)
#         self.pool  = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.5)
#         #add 1 layer mlp
#         self.fc1   = nn.Linear(32, 16) 
#         self.fc2   = nn.Linear(16, 1) 
#         #self.fc1   = nn.Linear(16, 1) # Changed in_features from 128 to 64
        

#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         #x = self.pool(F.selu(self.bn2(self.conv2(x))))
#         # Global Max Pooling
#         # x will have shape [batch_size, 64] after this line
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         x= self.fc1(x)
#         x = F.selu(x)
#         return self.fc2(x).squeeze(-1)



# ________________________________________________________#
# --------------------------------------------------------#
# ---------------------v2---------------------------------#
# # #--------------------------------------------------------#
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Assuming you want 5x5 kernels as per previous request
#         self.conv1 = nn.Conv2d(1, 16, 5, padding=1)  # 1 input channel, 32 output channels, 5x5 kernel
#         self.bn1   = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 16, 5, padding=1) #5 map 26 mrr 75
#         self.bn2   = nn.BatchNorm2d(16)
#         self.pool  = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1   = nn.Linear(16, 1) 
        

#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = self.pool(F.selu(self.bn2(self.conv2(x))))
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class KMaxPooling(nn.Module):
#     def __init__(self, k):
#         super().__init__()
#         self.k = k
#     def forward(self, x):
#         # x: (batch, channels, H, W)
#         b, c, h, w = x.size()
#         x = x.view(b, c, h*w)               # flatten spatial dims
#         topk, _ = x.topk(self.k, dim=-1)    # pick top k per channel
#         return topk.mean(dim=-1)            # avg → (batch, channels)

# class SimpleCNN(nn.Module):
#     def __init__(self, k=1):
#         super().__init__()
#         # conv‐block
#         self.conv1   = nn.Conv2d(1, 32, 5, padding=1)
#         self.bn1     = nn.BatchNorm2d(32)
#         self.pool    = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.5)

#         # k‐max pooling (replaces global max‐pool)
#         self.kmax    = KMaxPooling(k)

#         # MLP head
#         self.fc1 = nn.Linear(32, 1)

#     def forward(self, x):
#         x = self.pool(F.sigmoid(self.bn1(self.conv1(x))))
#         # x: (batch, 32, H', W')
#         x = self.kmax(x)           # → (batch, 32)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)



# #________________________________________________________#
# #--------------------------------------------------------#
# #---------------------v3---------------------------------#
# #--------------------------------------------------------#
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Assuming you want 5x5 kernels as per previous request
#         self.conv1 = nn.Conv2d(1, 16, 5, padding=1)  # 1 input channel, 32 output channels, 5x5 kernel
#         self.bn1   = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 16, 5, padding=1) #5 map 26 mrr 75
#         self.bn2   = nn.BatchNorm2d(16)
#         self.pool  = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1   = nn.Linear(16, 1) # Changed in_features from 128 to 64
        

#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = self.pool(F.selu(self.bn2(self.conv2(x))))
#         # Global Max Pooling
#         # x will have shape [batch_size, 64] after this line
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)



# import os
# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Assuming you want 5x5 kernels as per previous request
#         self.conv1 = nn.Conv2d(1,8, 5, padding=1)  # 1 input channel, 32 output channels, 5x5 kernel
#         self.bn1   = nn.BatchNorm2d(8)
#         self.conv2 = nn.Conv2d(8, 8, 5, padding=1) #5 map 26 mrr 75
#         self.bn2   = nn.BatchNorm2d(8)
#         self.pool  = nn.MaxPool2d(2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc1   = nn.Linear(8, 1) # Changed in_features from 128 to 64
        

#     def forward(self, x):
#         x = self.pool(F.selu(self.bn1(self.conv1(x))))
#         x = self.pool(F.selu(self.bn2(self.conv2(x))))
#         # Global Max Pooling
#         # x will have shape [batch_size, 64] after this line
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#         x = self.dropout(x)
#         return self.fc1(x).squeeze(-1)

# import json
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self, initial_channels=16, final_fc_channels=128, dropout_rate=0.3):
#         super().__init__()
        
#         # Block 1
#         self.conv1a = nn.Conv2d(1, initial_channels, kernel_size=3, padding=1)
#         self.bn1a = nn.BatchNorm2d(initial_channels)
#         self.conv1b = nn.Conv2d(initial_channels, initial_channels, kernel_size=3, padding=1)
#         self.bn1b = nn.BatchNorm2d(initial_channels)
#         self.pool1 = nn.MaxPool2d(2) # Halves spatial dimensions, e.g., 32x32 -> 16x16

#         # Block 2
#         ch_in_b2 = initial_channels
#         ch_out_b2 = initial_channels * 2
#         self.conv2a = nn.Conv2d(ch_in_b2, ch_out_b2, kernel_size=3, padding=1)
#         self.bn2a = nn.BatchNorm2d(ch_out_b2)
#         self.conv2b = nn.Conv2d(ch_out_b2, ch_out_b2, kernel_size=3, padding=1)
#         self.bn2b = nn.BatchNorm2d(ch_out_b2)
#         self.pool2 = nn.MaxPool2d(2) # e.g., 16x16 -> 8x8

#         # Block 3
#         ch_in_b3 = ch_out_b2
#         ch_out_b3 = initial_channels * 4
#         self.conv3a = nn.Conv2d(ch_in_b3, ch_out_b3, kernel_size=3, padding=1)
#         self.bn3a = nn.BatchNorm2d(ch_out_b3)
#         self.conv3b = nn.Conv2d(ch_out_b3, ch_out_b3, kernel_size=3, padding=1)
#         self.bn3b = nn.BatchNorm2d(ch_out_b3)
#         self.pool3 = nn.MaxPool2d(2) # e.g., 8x8 -> 4x4

#         # Block 4
#         ch_in_b4 = ch_out_b3
#         # Ensure final_fc_channels is used as the output of the last conv block
#         # This makes self.final_conv_channels consistent with final_fc_channels
#         self.final_conv_channels = final_fc_channels 
#         self.conv4a = nn.Conv2d(ch_in_b4, self.final_conv_channels, kernel_size=3, padding=1)
#         self.bn4a = nn.BatchNorm2d(self.final_conv_channels)
#         self.conv4b = nn.Conv2d(self.final_conv_channels, self.final_conv_channels, kernel_size=3, padding=1)
#         self.bn4b = nn.BatchNorm2d(self.final_conv_channels)
#         self.pool4 = nn.MaxPool2d(2) # e.g., 4x4 -> 2x2
        
#         self.dropout = nn.Dropout(dropout_rate)
#         # The input to fc1 will be the number of channels after the last conv block and global pooling
#         self.fc1 = nn.Linear(self.final_conv_channels, 1)

#     def forward(self, x):
#         # Block 1
#         x = F.selu(self.bn1a(self.conv1a(x)))
#         x = F.selu(self.bn1b(self.conv1b(x)))
#         x = self.pool1(x)

#         # Block 2
#         x = F.selu(self.bn2a(self.conv2a(x)))
#         x = F.selu(self.bn2b(self.conv2b(x)))
#         x = self.pool2(x)

#         # Block 3
#         x = F.selu(self.bn3a(self.conv3a(x)))
#         x = F.selu(self.bn3b(self.conv3b(x)))
#         x = self.pool3(x)

#         # Block 4
#         x = F.selu(self.bn4a(self.conv4a(x)))
#         x = F.selu(self.bn4b(self.conv4b(x)))
#         x = self.pool4(x)

#         # Global Max Pooling
#         # After 4 pooling layers, if input was HxW, it's now H/16 x W/16
#         # For example, if input is 32x32, after 4 pools, it's 2x2.
#         # adaptive_max_pool2d ensures output is (batch_size, self.final_conv_channels, 1, 1)
#         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1) 
        
#         x = self.dropout(x)
#         x = self.fc1(x).squeeze(-1) # Squeeze the last dimension for BCEWithLogitsLoss
#         return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 5x5 kernels, 16 channels, matching the checkpoint
        self.conv1 = nn.Conv2d(1, 16, 5, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        # Single FC layer mapping 16 features to 1, matching the checkpoint
        self.fc1   = nn.Linear(16, 1) 
        
    def forward(self, x):
        x = self.pool(F.selu(self.bn1(self.conv1(x))))
        x = self.pool(F.selu(self.bn2(self.conv2(x))))
        x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc1(x).squeeze(-1)