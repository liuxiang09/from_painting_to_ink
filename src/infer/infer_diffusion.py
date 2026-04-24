import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from case9.models.diffusion.condition_encoder import ConditionEncoder
from case9.models.diffusion.noise_unet import NoiseUNet


def load_models(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cond_encoder = ConditionEncoder().to(device)
    noise_unet = NoiseUNet().to(device)
    cond_encoder.load_state_dict(ckpt["cond_encoder"])
    noise_unet.load_state_dict(ckpt["noise_unet"])
    cond_encoder.eval()
    noise_unet.eval()
    alpha = ckpt["alpha"].to(device)
    alpha_bar = ckpt["alpha_bar"].to(device)
    return cond_encoder, noise_unet, alpha, alpha_bar


def reverse_step(x_t: torch.Tensor, pred_noise: torch.Tensor, t: int, alpha: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    a_t = alpha[t].view(1, 1, 1, 1)
    ab_t = alpha_bar[t].view(1, 1, 1, 1)
    x0 = (x_t - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t + 1e-8)

    prev_ab = alpha_bar[max(t - 1, 0)].view(1, 1, 1, 1)
    mean = torch.sqrt(prev_ab) * x0 + torch.sqrt(1 - prev_ab) * pred_noise

    if t > 0:
        z = torch.randn_like(x_t)
        sigma = torch.sqrt(((1 - prev_ab) / (1 - ab_t + 1e-8)) * (1 - a_t))
        out = mean + sigma * z
    else:
        out = mean

    return out.clamp(-1, 1)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/diffusion/infer")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cond_encoder, noise_unet, alpha, alpha_bar = load_models(args.checkpoint, device)

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with Image.open(args.content) as im:
        content = tfm(im.convert("RGB")).unsqueeze(0).to(device)
    with Image.open(args.style) as im:
        style = tfm(im.convert("RGB")).unsqueeze(0).to(device)

    content_emb = cond_encoder(content)
    style_emb = cond_encoder(style)

    x_t = torch.randn_like(content)
    max_steps = min(args.steps, int(alpha.shape[0]))
    with torch.no_grad():
        for t in reversed(range(max_steps)):
            t_tensor = torch.tensor([t], device=device)
            pred_noise = noise_unet(x_t, t_tensor, content_emb, style_emb)
            x_t = reverse_step(x_t, pred_noise, t, alpha, alpha_bar)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image((x_t[0] + 1.0) / 2.0, out_dir / "diffusion_ink.png")
    print(f"saved to {out_dir / 'diffusion_ink.png'}")


if __name__ == "__main__":
    main()
