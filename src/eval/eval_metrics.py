import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images(folder: Path, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
    ])
    imgs = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            with Image.open(p) as im:
                imgs.append(tfm(im.convert("RGB")))
    if not imgs:
        raise ValueError(f"No images under {folder}")
    return torch.stack(imgs)


def channel_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=(0, 2, 3))
    var = x.var(dim=(0, 2, 3), unbiased=False)
    return mean, var


def simple_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    m1, v1 = channel_stats(real)
    m2, v2 = channel_stats(fake)
    mean_dist = torch.sum((m1 - m2) ** 2)
    covmean = torch.sqrt(v1 * v2 + 1e-8)
    fid = mean_dist + torch.sum(v1 + v2 - 2 * covmean)
    return float(fid.item())


def content_l1(content: torch.Tensor, stylized: torch.Tensor) -> float:
    n = min(content.shape[0], stylized.shape[0])
    return float(torch.mean(torch.abs(content[:n] - stylized[:n])).item())


def gram(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    f = x.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def style_distance(style_ref: torch.Tensor, stylized: torch.Tensor) -> float:
    n = min(style_ref.shape[0], stylized.shape[0])
    g1 = gram(style_ref[:n])
    g2 = gram(stylized[:n])
    return float(torch.mean(torch.abs(g1 - g2)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style-ref", type=str, required=True)
    parser.add_argument("--cyclegan", type=str, required=True)
    parser.add_argument("--diffusion", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output", type=str, default="outputs/eval/metrics.csv")
    args = parser.parse_args()

    content = load_images(Path(args.content), args.image_size)
    style_ref = load_images(Path(args.style_ref), args.image_size)
    cyc = load_images(Path(args.cyclegan), args.image_size)
    diff = load_images(Path(args.diffusion), args.image_size)

    rows = [
        {
            "model": "CycleGAN",
            "content_l1": content_l1(content, cyc),
            "style_distance": style_distance(style_ref, cyc),
            "simple_fid_vs_style": simple_fid(style_ref, cyc),
        },
        {
            "model": "Diffusion",
            "content_l1": content_l1(content, diff),
            "style_distance": style_distance(style_ref, diff),
            "simple_fid_vs_style": simple_fid(style_ref, diff),
        },
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "content_l1", "style_distance", "simple_fid_vs_style"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved metrics to {out_path}")


if __name__ == "__main__":
    main()
