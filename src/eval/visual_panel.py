import argparse
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in VALID_EXT]


def load_fit(path: Path, size: int) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.fit(im.convert("RGB"), (size, size), method=Image.BICUBIC)


def draw_title(img: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 30), color=(255, 255, 255))
    canvas.paste(img, (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(0, 0, 0))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--cyclegan", type=str, required=True)
    parser.add_argument("--diffusion", type=str, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs/eval/panels")
    args = parser.parse_args()

    content_paths = list_images(Path(args.content))[: args.max_items]
    style_paths = list_images(Path(args.style))
    cyc_paths = list_images(Path(args.cyclegan))
    diff_paths = list_images(Path(args.diffusion))

    n = min(len(content_paths), len(style_paths), len(cyc_paths), len(diff_paths))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        c = draw_title(load_fit(content_paths[i], args.size), "Input")
        s = draw_title(load_fit(style_paths[i], args.size), "Style")
        g = draw_title(load_fit(cyc_paths[i], args.size), "CycleGAN")
        d = draw_title(load_fit(diff_paths[i], args.size), "Diffusion")

        panel = Image.new("RGB", (c.width * 4, c.height))
        panel.paste(c, (0, 0))
        panel.paste(s, (c.width, 0))
        panel.paste(g, (c.width * 2, 0))
        panel.paste(d, (c.width * 3, 0))

        panel.save(out_dir / f"panel_{i:03d}.png")

    print(f"saved {n} panels to {out_dir}")


if __name__ == "__main__":
    main()
