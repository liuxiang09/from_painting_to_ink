import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from case9.models.cyclegan.generator_unet import UNetGenerator


def load_model(ckpt_path: str, device: torch.device) -> UNetGenerator:
    model = UNetGenerator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["g"])
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

    image_paths = [in_path] if in_path.is_file() else [p for p in in_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    with torch.no_grad():
        for p in image_paths:
            with Image.open(p) as im:
                x = tfm(im.convert("RGB")).unsqueeze(0).to(device)
            y = model(x)
            save_image((y[0] + 1.0) / 2.0, out_dir / f"{p.stem}_ink.png")

    print(f"saved {len(image_paths)} images to {out_dir}")


if __name__ == "__main__":
    main()
