from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features[:35]
        except Exception:
            vgg = models.vgg19(weights=None).features[:35]

        self.features = vgg.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        return self.features(x)


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()


class ArtisticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = VGGFeature()

    def content(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1(self.vgg(fake), self.vgg(real))

    def style(self, fake: torch.Tensor, style_ref: torch.Tensor) -> torch.Tensor:
        fake_feature = self.vgg(fake)
        style_feature = self.vgg(style_ref)
        return self.l1(gram_matrix(fake_feature), gram_matrix(style_feature))

    def tv(self, fake: torch.Tensor) -> torch.Tensor:
        return total_variation_loss(fake)
