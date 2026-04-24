import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True, use_norm: bool = True):
        super().__init__()
        if down:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        else:
            conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_norm)

        layers = [conv]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64):
        super().__init__()
        self.d1 = ConvBlock(in_ch, base, down=True, use_norm=False)
        self.d2 = ConvBlock(base, base * 2, down=True)
        self.d3 = ConvBlock(base * 2, base * 4, down=True)
        self.d4 = ConvBlock(base * 4, base * 8, down=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u1 = ConvBlock(base * 8, base * 8, down=False)
        self.u2 = ConvBlock(base * 16, base * 4, down=False)
        self.u3 = ConvBlock(base * 8, base * 2, down=False)
        self.u4 = ConvBlock(base * 4, base, down=False)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        b = self.bottleneck(d4)
        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d4], dim=1))
        u3 = self.u3(torch.cat([u2, d3], dim=1))
        u4 = self.u4(torch.cat([u3, d2], dim=1))
        out = self.out(torch.cat([u4, d1], dim=1))
        return out
