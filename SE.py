import torch
import torch.nn as nn

class SE(nn.Module):
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y