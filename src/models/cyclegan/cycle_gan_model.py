from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.cyclegan.image_pool import ImagePool
from src.models.cyclegan.losses import ArtisticLoss
from src.models.cyclegan.networks import GANLoss, define_D, define_G


class CycleGANModel:
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.netG_A = define_G(
            3,
            3,
            cfg["ngf"],
            cfg["netG"],
            cfg["norm"],
            cfg["use_dropout"],
            cfg["init_type"],
            cfg["init_gain"],
            device,
        )
        self.netG_B = define_G(
            3,
            3,
            cfg["ngf"],
            cfg["netG"],
            cfg["norm"],
            cfg["use_dropout"],
            cfg["init_type"],
            cfg["init_gain"],
            device,
        )
        self.netD_A = define_D(
            3,
            cfg["ndf"],
            cfg["netD"],
            cfg["n_layers_D"],
            cfg["norm"],
            cfg["init_type"],
            cfg["init_gain"],
            device,
        )
        self.netD_B = define_D(
            3,
            cfg["ndf"],
            cfg["netD"],
            cfg["n_layers_D"],
            cfg["norm"],
            cfg["init_type"],
            cfg["init_gain"],
            device,
        )

        self.fake_A_pool = ImagePool(cfg["pool_size"])
        self.fake_B_pool = ImagePool(cfg["pool_size"])

        self.criterion_gan = GANLoss(cfg["gan_mode"]).to(device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_art = ArtisticLoss().to(device)

        self.optimizer_G = torch.optim.Adam(
            list(self.netG_A.parameters()) + list(self.netG_B.parameters()),
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.netD_A.parameters()) + list(self.netD_B.parameters()),
            lr=cfg["lr"],
            betas=tuple(cfg["betas"]),
        )

        self.real_A: torch.Tensor | None = None
        self.real_B: torch.Tensor | None = None
        self.fake_A: torch.Tensor | None = None
        self.fake_B: torch.Tensor | None = None
        self.rec_A: torch.Tensor | None = None
        self.rec_B: torch.Tensor | None = None
        self.idt_A: torch.Tensor | None = None
        self.idt_B: torch.Tensor | None = None
        self.losses: dict[str, float] = {}

    def set_input(self, batch: dict) -> None:
        self.real_A = batch["photo"].to(self.device)
        self.real_B = batch["ink"].to(self.device)

    def forward(self) -> None:
        assert self.real_A is not None and self.real_B is not None
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        pred_real = netD(real)
        loss_D_real = self.criterion_gan(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterion_gan(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self) -> None:
        assert self.real_B is not None and self.fake_B is not None
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self) -> None:
        assert self.real_A is not None and self.fake_A is not None
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self) -> None:
        assert self.real_A is not None and self.real_B is not None
        assert self.fake_A is not None and self.fake_B is not None
        assert self.rec_A is not None and self.rec_B is not None

        lambda_idt = self.cfg["identity_weight"]
        lambda_cycle = self.cfg["cycle_weight"]
        lambda_content = self.cfg["content_weight"]
        lambda_style = self.cfg["style_weight"]
        lambda_tv = self.cfg["tv_weight"]

        self.idt_A = self.netG_A(self.real_B)
        self.idt_B = self.netG_B(self.real_A)
        self.loss_idt_A = self.criterion_identity(self.idt_A, self.real_B) * lambda_cycle * lambda_idt
        self.loss_idt_B = self.criterion_identity(self.idt_B, self.real_A) * lambda_cycle * lambda_idt

        self.loss_G_A = self.criterion_gan(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterion_gan(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * lambda_cycle
        self.loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * lambda_cycle
        self.loss_content = self.criterion_art.content(self.fake_B, self.real_A) * lambda_content
        self.loss_style = self.criterion_art.style(self.fake_B, self.real_B) * lambda_style
        self.loss_tv = self.criterion_art.tv(self.fake_B) * lambda_tv

        self.loss_G = (
            self.loss_G_A
            + self.loss_G_B
            + self.loss_cycle_A
            + self.loss_cycle_B
            + self.loss_idt_A
            + self.loss_idt_B
            + self.loss_content
            + self.loss_style
            + self.loss_tv
        )
        self.loss_G.backward()

    @staticmethod
    def set_requires_grad(nets: list[nn.Module], requires_grad: bool = False) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def optimize_parameters(self) -> None:
        self.forward()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

        self.losses = self.get_current_losses()

    def evaluate(self, batch: dict) -> dict[str, float]:
        self.set_input(batch)
        self.forward()

        with torch.no_grad():
            lambda_idt = self.cfg["identity_weight"]
            lambda_cycle = self.cfg["cycle_weight"]
            lambda_content = self.cfg["content_weight"]
            lambda_style = self.cfg["style_weight"]
            lambda_tv = self.cfg["tv_weight"]

            idt_A = self.netG_A(self.real_B)
            idt_B = self.netG_B(self.real_A)

            loss_idt_A = self.criterion_identity(idt_A, self.real_B) * lambda_cycle * lambda_idt
            loss_idt_B = self.criterion_identity(idt_B, self.real_A) * lambda_cycle * lambda_idt
            loss_G_A = self.criterion_gan(self.netD_A(self.fake_B), True)
            loss_G_B = self.criterion_gan(self.netD_B(self.fake_A), True)
            loss_cycle_A = self.criterion_cycle(self.rec_A, self.real_A) * lambda_cycle
            loss_cycle_B = self.criterion_cycle(self.rec_B, self.real_B) * lambda_cycle
            loss_content = self.criterion_art.content(self.fake_B, self.real_A) * lambda_content
            loss_style = self.criterion_art.style(self.fake_B, self.real_B) * lambda_style
            loss_tv = self.criterion_art.tv(self.fake_B) * lambda_tv
            loss_D_A = (self.criterion_gan(self.netD_A(self.real_B), True) + self.criterion_gan(self.netD_A(self.fake_B), False)) * 0.5
            loss_D_B = (self.criterion_gan(self.netD_B(self.real_A), True) + self.criterion_gan(self.netD_B(self.fake_A), False)) * 0.5

            total_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_content + loss_style + loss_tv
            return {
                "loss_G": total_G.item(),
                "loss_D_A": loss_D_A.item(),
                "loss_D_B": loss_D_B.item(),
                "loss_G_A": loss_G_A.item(),
                "loss_G_B": loss_G_B.item(),
                "loss_cycle_A": loss_cycle_A.item(),
                "loss_cycle_B": loss_cycle_B.item(),
                "loss_idt_A": loss_idt_A.item(),
                "loss_idt_B": loss_idt_B.item(),
                "loss_content": loss_content.item(),
                "loss_style": loss_style.item(),
                "loss_tv": loss_tv.item(),
            }

    def get_current_losses(self) -> dict[str, float]:
        return {
            "loss_G": self.loss_G.item(),
            "loss_D_A": self.loss_D_A.item(),
            "loss_D_B": self.loss_D_B.item(),
            "loss_G_A": self.loss_G_A.item(),
            "loss_G_B": self.loss_G_B.item(),
            "loss_cycle_A": self.loss_cycle_A.item(),
            "loss_cycle_B": self.loss_cycle_B.item(),
            "loss_idt_A": self.loss_idt_A.item(),
            "loss_idt_B": self.loss_idt_B.item(),
            "loss_content": self.loss_content.item(),
            "loss_style": self.loss_style.item(),
            "loss_tv": self.loss_tv.item(),
        }

    def train(self) -> None:
        self.netG_A.train()
        self.netG_B.train()
        self.netD_A.train()
        self.netD_B.train()

    def eval(self) -> None:
        self.netG_A.eval()
        self.netG_B.eval()
        self.netD_A.eval()
        self.netD_B.eval()

    def save_networks(self, epoch: int, ckpt_dir: Path, extra: dict[str, Any] | None = None) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "G_A": self.netG_A.state_dict(),
            "G_B": self.netG_B.state_dict(),
            "D_A": self.netD_A.state_dict(),
            "D_B": self.netD_B.state_dict(),
            "model_config": {
                "ngf": self.cfg["ngf"],
                "ndf": self.cfg["ndf"],
                "netG": self.cfg["netG"],
                "netD": self.cfg["netD"],
                "n_layers_D": self.cfg["n_layers_D"],
                "norm": self.cfg["norm"],
                "use_dropout": self.cfg["use_dropout"],
                "init_type": self.cfg["init_type"],
                "init_gain": self.cfg["init_gain"],
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, ckpt_dir / f"cyclegan_epoch_{epoch:03d}.pt")

