import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
