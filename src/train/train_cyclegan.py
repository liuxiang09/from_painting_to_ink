import argparse
import os
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from case9.data.dataset import UnpairedDataset
from case9.models.cyclegan.discriminator import PatchDiscriminator
from case9.models.cyclegan.generator_unet import UNetGenerator
from case9.models.cyclegan.losses import CycleLoss, tv_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["photo"].to(device), batch["ink"].to(device)


def save_sample(step: int, photo: torch.Tensor, fake_ink: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([photo[:4], fake_ink[:4]], dim=0)
    save_image((grid + 1.0) / 2.0, out_dir / f"sample_{step:07d}.png", nrow=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="case9/configs/cyclegan.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    ds = UnpairedDataset(
        photo_root=cfg["paths"]["photo_train"],
        ink_root=cfg["paths"]["ink_train"],
        image_size=cfg["image_size"],
        augment=True,
    )
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=True)

    g = UNetGenerator().to(device)
    f = UNetGenerator().to(device)
    d_ink = PatchDiscriminator().to(device)
    d_photo = PatchDiscriminator().to(device)

    loss_fn = CycleLoss().to(device)

    opt_g = Adam(list(g.parameters()) + list(f.parameters()), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    opt_d_ink = Adam(d_ink.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))
    opt_d_photo = Adam(d_photo.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]))

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    sample_dir = Path(cfg["paths"]["samples"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in dl:
            photo, ink = to_device(batch, device)

            fake_ink = g(photo)
            rec_photo = f(fake_ink)
            fake_photo = f(ink)
            rec_ink = g(fake_photo)

            id_photo = f(photo)
            id_ink = g(ink)

            opt_g.zero_grad()
            adv_g = loss_fn.adversarial(d_ink(fake_ink), True)
            adv_f = loss_fn.adversarial(d_photo(fake_photo), True)
            cyc = loss_fn.cycle(rec_photo, photo) + loss_fn.cycle(rec_ink, ink)
            ident = loss_fn.identity(id_photo, photo) + loss_fn.identity(id_ink, ink)
            content = loss_fn.content(fake_ink, photo)
            style = loss_fn.style(fake_ink, ink)
            smooth = tv_loss(fake_ink)

            loss_g = (
                adv_g
                + adv_f
                + cfg["cycle_weight"] * cyc
                + cfg["identity_weight"] * ident
                + cfg["content_weight"] * content
                + cfg["style_weight"] * style
                + cfg["tv_weight"] * smooth
            )
            loss_g.backward()
            opt_g.step()

            opt_d_ink.zero_grad()
            d_ink_real = loss_fn.adversarial(d_ink(ink), True)
            d_ink_fake = loss_fn.adversarial(d_ink(fake_ink.detach()), False)
            loss_d_ink = (d_ink_real + d_ink_fake) * 0.5
            loss_d_ink.backward()
            opt_d_ink.step()

            opt_d_photo.zero_grad()
            d_photo_real = loss_fn.adversarial(d_photo(photo), True)
            d_photo_fake = loss_fn.adversarial(d_photo(fake_photo.detach()), False)
            loss_d_photo = (d_photo_real + d_photo_fake) * 0.5
            loss_d_photo.backward()
            opt_d_photo.step()

            global_step += 1
            if global_step % cfg["sample_every"] == 0:
                save_sample(global_step, photo, fake_ink, sample_dir)

        if epoch % cfg["save_every"] == 0:
            torch.save({"g": g.state_dict(), "f": f.state_dict()}, ckpt_dir / f"cyclegan_epoch_{epoch:03d}.pt")

        print(
            f"epoch={epoch}/{cfg['epochs']} "
            f"loss_g={loss_g.item():.4f} "
            f"loss_d_ink={loss_d_ink.item():.4f} "
            f"loss_d_photo={loss_d_photo.item():.4f}"
        )


if __name__ == "__main__":
    main()
