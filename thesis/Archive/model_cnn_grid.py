import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, conv_blocks, hidden, kernel_size, padding, dropout, mlp_head):
        super().__init__()
        layers = []
        in_ch = 1
        for _ in range(conv_blocks):
            layers += [
                nn.Conv2d(in_ch, hidden, kernel_size, padding=padding),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.MaxPool2d(2),
            ]
            in_ch = hidden
        self.features    = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        head = []
        if mlp_head:
            head += [
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            ]
        else:
            head += [
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            ]
        self.head = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.head(x).squeeze(-1)