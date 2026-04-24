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


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean() + (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean()


class CycleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.vgg = VGGFeature()

    def adversarial(self, pred_fake: torch.Tensor, target_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred_fake) if target_real else torch.zeros_like(pred_fake)
        return self.mse(pred_fake, target)

    def cycle(self, rec: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1(rec, real)

    def identity(self, ident: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1(ident, real)

    def content(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1(self.vgg(fake), self.vgg(real))

    def style(self, fake: torch.Tensor, style_ref: torch.Tensor) -> torch.Tensor:
        fake_f = self.vgg(fake)
        style_f = self.vgg(style_ref)
        return self.l1(gram_matrix(fake_f), gram_matrix(style_f))
