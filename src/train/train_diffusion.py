import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from case9.data.dataset import UnpairedDataset
from case9.models.diffusion.condition_encoder import ConditionEncoder
from case9.models.diffusion.losses import DiffusionLoss
from case9.models.diffusion.noise_unet import NoiseUNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_beta_schedule(steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, steps, device=device)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise


def ddim_like_step(x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor, alpha: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a_t = alpha[t].view(-1, 1, 1, 1)
    ab_t = alpha_bar[t].view(-1, 1, 1, 1)
    x0_pred = (x_t - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t + 1e-8)
    mean = torch.sqrt(a_t) * x0_pred + torch.sqrt(1 - a_t) * pred_noise
    return mean.clamp(-1, 1)


def save_sample(step: int, photo: torch.Tensor, pred: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([photo[:4], pred[:4]], dim=0)
    save_image((grid + 1.0) / 2.0, out_dir / f"sample_{step:07d}.png", nrow=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="case9/configs/diffusion.yaml")
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

    cond_encoder = ConditionEncoder().to(device)
    noise_unet = NoiseUNet().to(device)
    loss_fn = DiffusionLoss().to(device)

    opt = AdamW(
        list(cond_encoder.parameters()) + list(noise_unet.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    beta = make_beta_schedule(cfg["timesteps"], device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    sample_dir = Path(cfg["paths"]["samples"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        for batch in dl:
            photo = batch["photo"].to(device)
            ink = batch["ink"].to(device)

            t = torch.randint(0, cfg["timesteps"], (photo.size(0),), device=device)
            noise = torch.randn_like(photo)
            x_t = q_sample(photo, t, noise, alpha_bar)

            content_emb = cond_encoder(photo)
            style_emb = cond_encoder(ink)

            pred_noise = noise_unet(x_t, t, content_emb, style_emb)
            pred_img = ddim_like_step(x_t, t, pred_noise, alpha, alpha_bar)

            noise_loss = loss_fn.noise(pred_noise, noise)
            content_loss = loss_fn.content(pred_img, photo)
            style_loss = loss_fn.style(pred_img, ink)

            loss = cfg["noise_weight"] * noise_loss + cfg["content_weight"] * content_loss + cfg["style_weight"] * style_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % cfg["sample_every"] == 0:
                save_sample(global_step, photo, pred_img, sample_dir)

        if epoch % cfg["save_every"] == 0:
            torch.save(
                {
                    "cond_encoder": cond_encoder.state_dict(),
                    "noise_unet": noise_unet.state_dict(),
                    "alpha": alpha.detach().cpu(),
                    "alpha_bar": alpha_bar.detach().cpu(),
                },
                ckpt_dir / f"diffusion_epoch_{epoch:03d}.pt",
            )

        print(
            f"epoch={epoch}/{cfg['epochs']} "
            f"loss={loss.item():.4f} noise={noise_loss.item():.4f} "
            f"content={content_loss.item():.4f} style={style_loss.item():.4f}"
        )


if __name__ == "__main__":
    main()
