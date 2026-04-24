import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.emb_proj = nn.Linear(emb_dim, out_ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        e = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + e
        h = self.conv2(h)
        h = self.bn2(h)
        return torch.relu(h + self.skip(x))


class NoiseUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64, emb_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(emb_dim)
        self.cond_mlp = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))

        self.down1 = ResBlock(in_ch, base, emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(base, base * 2, emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.mid = ResBlock(base * 2, base * 4, emb_dim)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ResBlock(base * 4, base * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ResBlock(base * 2, base, emb_dim)
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, content_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)
        cemb = self.cond_mlp(torch.cat([content_emb, style_emb], dim=1))
        emb = temb + cemb

        d1 = self.down1(x, emb)
        d2 = self.down2(self.pool1(d1), emb)
        m = self.mid(self.pool2(d2), emb)

        u2 = self.up2(m)
        u2 = self.dec2(torch.cat([u2, d2], dim=1), emb)
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1), emb)
        return self.out(u1)
