import torch
from torch import nn


class ConditionEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, emb_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)
