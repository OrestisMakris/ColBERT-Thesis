# 80rrrrr,lcdddddddd,
import torch
import torch.nn as nn
import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 1 input channel, 16 output channels, 5x5 kernel
#         self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 16, 5, padding=1)
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

import torch
import torch.nn as nn
import torch.nn.functional as F
# #ldr
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 1 input channel, 16 output channels, 5x5 kernel
#         self.conv1 = nn.Conv2d(1, 32, 6, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 16, 6, padding=1)
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

# arcade
# class SimpleCNN(nn.Module):
#      def __init__(self):
#          super().__init__()  
#          # 1 input channel, 16 output channels, 5x5 kernel
#          self.conv1 = nn.Conv2d(1, 32, 6, padding=1)
#          self.bn1   = nn.BatchNorm2d(32)
#          self.pool  = nn.MaxPool2d(2)
#          self.dropout = nn.Dropout(0.3)
#          self.fc1   = nn.Linear(32, 1)

#      def forward(self, x):
#          x = self.pool(F.selu(self.bn1(self.conv1(x))))
#          x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#          x = self.dropout(x)
#          return self.fc1(x).squeeze(-1)


#xx #usurum
# class SimpleCNN(nn.Module):
#      def __init__(self):
#          super().__init__()
#          # 1 input channel, 16 output channels, 5x5 kernel
#          self.conv1 = nn.Conv2d(1, 16, 6, padding=1)
#          self.bn1   = nn.BatchNorm2d(16)
#          self.pool  = nn.MaxPool2d(2)
#          self.dropout = nn.Dropout(0.3)
#          self.fc1   = nn.Linear( 16, 1)

#      def forward(self, x):
#          x = self.pool(F.selu(self.bn1(self.conv1(x))))
#          x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
#          x = self.dropout(x)
#          return self.fc1(x).squeeze(-1)

#triliza
class SimpleCNN(nn.Module):
     def __init__(self):
         super().__init__()
         # 1 input channel, 16 output channels, 5x5 kernel
         self.conv1 = nn.Conv2d(1,8, 6, padding=1)
         self.bn1   = nn.BatchNorm2d(8)
         self.pool  = nn.MaxPool2d(2)
         self.dropout = nn.Dropout(0.3)
         self.fc1   = nn.Linear( 8, 1)

     def forward(self, x):
         x = self.pool(F.selu(self.bn1(self.conv1(x))))
         x = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
         x = self.dropout(x)
         return self.fc1(x).squeeze(-1)

         

    