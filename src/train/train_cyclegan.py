import argparse
import random
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.dataset import UnpairedDataset
from src.models.cyclegan.cycle_gan_model import CycleGANModel
from src.train.monitor import mean_metric_dict, plot_metric_groups, save_history_csv

PROGRESS_NCOLS = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_sample(step: int, photo: torch.Tensor, fake_ink: torch.Tensor, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.cat([photo[:4], fake_ink[:4]], dim=0)
    save_image((grid + 1.0) / 2.0, out_dir / f"sample_{step:07d}.png", nrow=4)


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def init_metric_sums() -> dict[str, float]:
    return {
        "loss_G": 0.0,
        "loss_D_A": 0.0,
        "loss_D_B": 0.0,
        "loss_G_A": 0.0,
        "loss_G_B": 0.0,
        "loss_cycle_A": 0.0,
        "loss_cycle_B": 0.0,
        "loss_idt_A": 0.0,
        "loss_idt_B": 0.0,
        "loss_content": 0.0,
        "loss_style": 0.0,
        "loss_tv": 0.0,
    }


def accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float]) -> None:
    for key in metric_sums:
        metric_sums[key] += metrics[key]


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
    parser.add_argument("--config", type=str, default="src/configs/cyclegan.yaml")
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
        augment=True,
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)

    model = CycleGANModel(cfg, device)
    model.train()

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

        checkpoint = torch.load(resume_path, map_location=device)
        model.netG_A.load_state_dict(checkpoint["G_A"])
        model.netG_B.load_state_dict(checkpoint["G_B"])
        model.netD_A.load_state_dict(checkpoint["D_A"])
        model.netD_B.load_state_dict(checkpoint["D_B"])

        if "optimizer_G" in checkpoint:
            model.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        if "optimizer_D" in checkpoint:
            model.optimizer_D.load_state_dict(checkpoint["optimizer_D"])

        ckpt_epoch = int(checkpoint.get("epoch", 0))
        target_epoch = int(cfg["epochs"])
        if ckpt_epoch >= target_epoch:
            print(f"checkpoint epoch {ckpt_epoch} already reaches target epochs {target_epoch}; nothing to train")
            return

        start_epoch = ckpt_epoch + 1
        global_step = int(checkpoint.get("global_step", ckpt_epoch * len(train_dl)))
        print(f"resuming CycleGAN from epoch {start_epoch}/{target_epoch}, global_step={global_step}, checkpoint={resume_path}")

    history = load_history_csv(metrics_file)
    history = [entry for entry in history if int(entry["epoch"]) < start_epoch]

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        metric_sums = init_metric_sums()
        progress = tqdm(
            train_dl,
            desc=f"CycleGAN Epoch {epoch}/{cfg['epochs']}",
            ncols=PROGRESS_NCOLS,
            dynamic_ncols=False,
            leave=True,
        )

        for batch in progress:
            model.set_input(batch)
            model.optimize_parameters()
            metrics = model.get_current_losses()
            accumulate_metrics(metric_sums, metrics)

            progress.set_postfix(
                {
                    "G": f"{metrics['loss_G']:.4f}",
                    "D_A": f"{metrics['loss_D_A']:.4f}",
                    "D_B": f"{metrics['loss_D_B']:.4f}",
                },
                refresh=False,
            )

            global_step += 1
            if global_step % cfg["sample_every"] == 0:
                save_sample(global_step, model.real_A, model.fake_B, sample_dir)

        train_metrics = mean_metric_dict(metric_sums, len(train_dl))

        history.append(
            {
                "epoch": epoch,
                "loss_G": train_metrics["loss_G"],
                "loss_D_A": train_metrics["loss_D_A"],
                "loss_D_B": train_metrics["loss_D_B"],
                "loss_G_A": train_metrics["loss_G_A"],
                "loss_G_B": train_metrics["loss_G_B"],
                "cycle_A": train_metrics["loss_cycle_A"],
                "cycle_B": train_metrics["loss_cycle_B"],
                "identity_A": train_metrics["loss_idt_A"],
                "identity_B": train_metrics["loss_idt_B"],
                "content": train_metrics["loss_content"],
                "style": train_metrics["loss_style"],
                "tv": train_metrics["loss_tv"],
            }
        )

        save_history_csv(history, metrics_file)
        plot_metric_groups(
            history,
            metrics_dir,
            [
                ("loss_curves.png", "CycleGAN Training Loss", ["loss_G", "loss_D_A", "loss_D_B"]),
                ("component_curves.png", "CycleGAN Training Components", ["cycle_A", "cycle_B", "identity_A", "identity_B", "content", "style", "tv"]),
            ],
        )

        if epoch % cfg["save_every"] == 0:
            model.save_networks(
                epoch,
                ckpt_dir,
                extra={
                    "global_step": global_step,
                    "optimizer_G": model.optimizer_G.state_dict(),
                    "optimizer_D": model.optimizer_D.state_dict(),
                },
            )

        print(
            f"epoch={epoch}/{cfg['epochs']} "
            f"loss_G={train_metrics['loss_G']:.4f} "
            f"content={train_metrics['loss_content']:.4f} "
            f"style={train_metrics['loss_style']:.4f}"
        )


if __name__ == "__main__":
    main()
