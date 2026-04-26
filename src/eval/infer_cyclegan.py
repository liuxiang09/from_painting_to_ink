import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.models.cyclegan.networks import define_G


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["model_config"]
    model = define_G(
        3,
        3,
        model_cfg["ngf"],
        model_cfg["netG"],
        model_cfg["norm"],
        model_cfg["use_dropout"],
        model_cfg["init_type"],
        model_cfg["init_gain"],
        device,
    )
    model.load_state_dict(ckpt["G_A"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/cyclegan/infer")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [in_path] if in_path.is_file() else [p for p in sorted(in_path.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    with torch.no_grad():
        for p in image_paths:
            with Image.open(p) as im:
                x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
            y = model(x)
            save_image((y[0] + 1.0) / 2.0, out_dir / f"{p.stem}_ink.png")

    print(f"saved {len(image_paths)} images to {out_dir}")


if __name__ == "__main__":
    main()
