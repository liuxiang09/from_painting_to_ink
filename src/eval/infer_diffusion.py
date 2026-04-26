import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.models.diffusion.stable_diffusion import StableDiffusionStyleTransfer


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    with Image.open(path) as im:
        return tfm(im.convert("RGB")).unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/diffusion/infer")
    parser.add_argument("--pretrained-model", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--style-weight", type=float, default=0.75)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" and args.fp16 else torch.float32
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    pretrained_model = ckpt.get("pretrained_model", args.pretrained_model)

    model = StableDiffusionStyleTransfer(
        pretrained_model=pretrained_model,
        style_weight=args.style_weight,
        train_unet=False,
        train_vae=False,
        torch_dtype=dtype,
    ).to(device)
    model.load_training_checkpoint(args.checkpoint, map_location=device)

    content = load_image(args.content, args.image_size, device)
    style = load_image(args.style, args.image_size, device)

    with torch.no_grad():
        pred = model.generate(
            content=content,
            style=style,
            num_inference_steps=args.steps,
            strength=args.strength,
            style_weight=args.style_weight,
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image((pred[0] + 1.0) / 2.0, out_dir / "diffusion_ink.png")
    print(f"saved to {out_dir / 'diffusion_ink.png'}")


if __name__ == "__main__":
    main()
