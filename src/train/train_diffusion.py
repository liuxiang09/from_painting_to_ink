import argparse
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import UnpairedDataset
from src.models.diffusion.losses import DiffusionLoss
from src.models.diffusion.stable_diffusion import StableDiffusionStyleTransfer
from src.train.monitor import mean_metric_dict, save_history_csv

PROGRESS_NCOLS = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_sample(step: int, photo: torch.Tensor, pred: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([photo[:4], pred[:4]], dim=0)
    save_image((grid + 1.0) / 2.0, out_dir / f"sample_{step:07d}.png", nrow=4)


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def init_metric_sums() -> dict[str, float]:
    return {
        "loss": 0.0,
        "noise": 0.0,
        "content": 0.0,
        "style": 0.0,
    }


def accumulate_metrics(metric_sums: dict[str, float], batch_metrics: dict[str, float]) -> None:
    for key in metric_sums:
        metric_sums[key] += batch_metrics[key]


def load_history_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    history: list[dict[str, float]] = []
    for row in rows:
        history.append({key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()})
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/diffusion.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    loader_kwargs = {
        "batch_size": cfg["batch_size"],
        "num_workers": cfg["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }

    train_ds = UnpairedDataset(
        photo_root=cfg["paths"]["photo_train"],
        ink_root=cfg["paths"]["ink_train"],
        image_size=cfg["image_size"],
        augment=cfg.get("augment", True),
        photo_color_jitter_strength=float(cfg.get("photo_color_jitter", 0.05)),
        ink_color_jitter_strength=float(cfg.get("ink_color_jitter", 0.05)),
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)

    model_cfg = cfg.get("model", {})
    use_amp = device.type == "cuda" and model_cfg.get("fp16", False)
    dtype = torch.float16 if use_amp else torch.float32
    model = StableDiffusionStyleTransfer(
        pretrained_model=model_cfg.get("pretrained_model", "CompVis/stable-diffusion-v1-4"),
        style_weight=cfg["style_weight"],
        train_unet=model_cfg.get("train_unet", True),
        train_vae=model_cfg.get("train_vae", False),
        tokens_per_image=model_cfg.get("tokens_per_image", 32),
        torch_dtype=dtype,
    ).to(device)
    if use_amp:
        model.unet.float()
        model.condition_projector.float()
    loss_fn = DiffusionLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    trainable_params = [param for param in model.trainable_parameters() if param.requires_grad]
    total_trainable = sum(param.numel() for param in trainable_params)
    unet_trainable = sum(param.numel() for param in model.unet.parameters() if param.requires_grad)
    print(f"diffusion mode={model.unet_mode} trainable_params={total_trainable} unet_trainable={unet_trainable}")

    opt = AdamW(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    sample_dir = Path(cfg["paths"]["samples"])
    metrics_dir = ckpt_dir.parent / "metrics"
    metrics_file = metrics_dir / "history.csv"

    if args.resume is None:
        reset_output_dir(ckpt_dir)
        reset_output_dir(sample_dir)
        reset_output_dir(metrics_dir)

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

        checkpoint = model.load_training_checkpoint(resume_path, map_location=device)
        if "opt" in checkpoint:
            try:
                opt.load_state_dict(checkpoint["opt"])
            except ValueError as exc:
                print(f"warning: optimizer state incompatible with current trainable params, skip restoring optimizer. detail: {exc}")
        if "scaler" in checkpoint and use_amp:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except Exception as exc:
                print(f"warning: scaler state incompatible, skip restoring scaler. detail: {exc}")

        ckpt_epoch = int(checkpoint.get("epoch", 0))
        target_epoch = int(cfg["epochs"])
        if ckpt_epoch >= target_epoch:
            print(f"checkpoint epoch {ckpt_epoch} already reaches target epochs {target_epoch}; nothing to train")
            return

        start_epoch = ckpt_epoch + 1
        global_step = int(checkpoint.get("global_step", ckpt_epoch * len(train_dl)))
        print(f"resuming diffusion from epoch {start_epoch}/{target_epoch}, global_step={global_step}, checkpoint={resume_path}")

    history = load_history_csv(metrics_file)
    history = [entry for entry in history if int(entry["epoch"]) < start_epoch]

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        metric_sums = init_metric_sums()
        progress = tqdm(
            train_dl,
            desc=f"Diffusion Epoch {epoch}/{cfg['epochs']}",
            ncols=PROGRESS_NCOLS,
            dynamic_ncols=False,
            leave=True,
        )

        for batch in progress:
            model.train()
            photo = batch["photo"].to(device)
            ink = batch["ink"].to(device)

            t_min = max(0, int(cfg.get("timestep_min", 0)))
            t_max = min(int(cfg["timesteps"]) - 1, int(cfg.get("timestep_max", int(cfg["timesteps"]) - 1)))
            if t_max < t_min:
                raise ValueError(f"invalid timestep range: timestep_min={t_min}, timestep_max={t_max}")
            t = torch.randint(t_min, t_max + 1, (photo.size(0),), device=device)
            with torch.no_grad():
                latents = model.encode_images(photo)
            noise = torch.randn_like(latents)
            noisy_latents = model.add_noise(latents, noise, t)

            amp_context = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext()
            with amp_context:
                pred_noise = model.predict_noise(noisy_latents, t, photo, ink, style_weight=cfg["style_weight"])
                pred_latents = model.predict_x0(noisy_latents, pred_noise, t)
                pred_img = model.decode_latents(pred_latents)

                noise_loss = loss_fn.noise(pred_noise, noise)
                content_loss = loss_fn.content(pred_img, photo)
                style_loss = loss_fn.style(pred_img, ink)
                loss = cfg["noise_weight"] * noise_loss + cfg["content_weight"] * content_loss + cfg["style_weight"] * style_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            accumulate_metrics(
                metric_sums,
                {
                    "loss": loss.item(),
                    "noise": noise_loss.item(),
                    "content": content_loss.item(),
                    "style": style_loss.item(),
                },
            )

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "noise": f"{noise_loss.item():.4f}",
                    "style": f"{style_loss.item():.4f}",
                },
                refresh=False,
            )

            global_step += 1
            if global_step % cfg["sample_every"] == 0:
                save_sample(global_step, photo, pred_img, sample_dir)

        train_metrics = mean_metric_dict(metric_sums, len(train_dl))

        history.append(
            {
                "epoch": epoch,
                "loss": train_metrics["loss"],
                "noise": train_metrics["noise"],
                "content": train_metrics["content"],
                "style": train_metrics["style"],
            }
        )

        save_history_csv(history, metrics_file)

        if epoch % cfg["save_every"] == 0:
            model.save_training_checkpoint(
                ckpt_dir / f"diffusion_epoch_{epoch:03d}.pt",
                extra={
                    "epoch": epoch,
                    "global_step": global_step,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "unet_mode": model.unet_mode,
                    "pretrained_model": model_cfg.get("pretrained_model", "CompVis/stable-diffusion-v1-4"),
                    "style_weight": cfg["style_weight"],
                    "timesteps": cfg["timesteps"],
                },
            )

        print(
            f"epoch={epoch}/{cfg['epochs']} "
            f"loss={train_metrics['loss']:.4f} "
            f"content={train_metrics['content']:.4f} "
            f"style={train_metrics['style']:.4f}"
        )


if __name__ == "__main__":
    main()
