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
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        return self.features(x)


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    gram = feat @ feat.transpose(1, 2)
    return gram / (c * h * w)


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.vgg = VGGFeature()

    def noise(self, pred_noise: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.mse(pred_noise, noise)

    def content(self, pred_img: torch.Tensor, real_img: torch.Tensor) -> torch.Tensor:
        return self.l1(self.vgg(pred_img), self.vgg(real_img))

    def style(self, pred_img: torch.Tensor, style_ref: torch.Tensor) -> torch.Tensor:
        pred_f = self.vgg(pred_img)
        style_f = self.vgg(style_ref)
        return self.l1(gram_matrix(pred_f), gram_matrix(style_f))
