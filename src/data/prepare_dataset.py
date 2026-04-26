from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable

try:
    from PIL import Image, ImageDraw, ImageOps

    PIL_AVAILABLE = True
except ModuleNotFoundError:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageOps = None  # type: ignore
    PIL_AVAILABLE = False

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INK_CATEGORY_ORDER = ("gong_bi", "shui_mo", "shan_shui", "guo_hua", "yuan_ti", "other")
PROGRESS_NCOLS = 100
CLIP_POSITIVE_PROMPTS = [
    "a landscape photo",
    "a nature landscape",
    "a mountain landscape",
    "a river in nature",
    "a lake in nature",
    "a forest landscape",
    "a scenic outdoor view",
    "a natural scenery photograph",
    "a wilderness landscape",
    "a beach landscape",
]
CLIP_NEGATIVE_PROMPTS = [
    "a portrait photo of a person",
    "a close-up photo of a person",
    "a photo of an animal",
    "a dog or cat photo",
    "a sports scene",
    "a street scene with people",
    "an indoor scene",
    "a selfie or portrait",
    "a person standing for the camera",
    "a photo focused on a human subject",
]


@dataclass
class ImageRecord:
    source: str
    image_path: Path
    caption: Optional[str] = None
    category: Optional[str] = None
    score: float = 0.0


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"Missing directory: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory: {root}")
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])


def infer_ink_category(path: Path) -> str:
    stem = path.stem.lower()
    if "gong_bi" in stem:
        return "gong_bi"
    if "shui_mo" in stem:
        return "shui_mo"
    if "shan_shui" in stem:
        return "shan_shui"
    if "guo_hua" in stem:
        return "guo_hua"
    if "yuan_ti" in stem:
        return "yuan_ti"
    return "other"


def load_ink_records(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    image_paths = collect_images(root)
    for image_path in tqdm(
        image_paths,
        desc="Scan ink_src",
        ncols=PROGRESS_NCOLS,
        dynamic_ncols=False,
        leave=False,
    ):
        records.append(
            ImageRecord(
                source="ink_src",
                image_path=image_path,
                category=infer_ink_category(image_path),
            )
        )
    if not records:
        raise ValueError(f"No images found under {root}")
    return records


def load_flickr8k_caption_map(root: Path) -> dict[str, list[str]]:
    caption_path = root / "captions.txt"
    if not caption_path.exists():
        raise ValueError(f"Missing Flickr8k caption file: {caption_path}")

    captions_by_image: dict[str, list[str]] = {}
    with caption_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = (row.get("image") or "").strip()
            caption = (row.get("caption") or "").strip()
            if image_name and caption:
                captions_by_image.setdefault(image_name, []).append(caption)
    return captions_by_image


def load_flickr8k_image_paths(root: Path) -> list[Path]:
    image_root = root / "Images"
    if not image_root.exists():
        raise ValueError(f"Missing Flickr8k image folder: {image_root}")
    return collect_images(image_root)


def build_clip_ranker(model_name: str):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CLIP selection needs 'torch', 'transformers', and 'Pillow'. "
            "Install them before using CLIP-based dataset filtering."
        ) from exc

    if not PIL_AVAILABLE:
        raise ModuleNotFoundError("CLIP selection needs Pillow installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = os.getenv("HF_TOKEN") or None
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CLIP model '{model_name}' with safetensors. "
            "Please ensure the model provides safetensors weights, and that "
            "'transformers' plus 'safetensors' are installed."
        ) from exc

    processor = CLIPProcessor.from_pretrained(model_name, token=token)
    model.to(device)
    model.eval()
    prompts = CLIP_POSITIVE_PROMPTS + CLIP_NEGATIVE_PROMPTS
    positive_count = len(CLIP_POSITIVE_PROMPTS)

    def score_image(image_path: Path) -> float:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        with torch.no_grad():
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            positive_score = logits[:positive_count].mean().item()
            negative_score = logits[positive_count:].mean().item()
            return positive_score - negative_score

    return score_image


def load_flickr8k_landscape_records_with_clip(root: Path, target_count: int, clip_model_name: str) -> list[ImageRecord]:
    captions_by_image = load_flickr8k_caption_map(root)
    image_paths = load_flickr8k_image_paths(root)
    score_image = build_clip_ranker(clip_model_name)

    scored_records: list[ImageRecord] = []
    for image_path in tqdm(
        image_paths,
        desc="CLIP score Flickr8k",
        ncols=PROGRESS_NCOLS,
        dynamic_ncols=False,
        leave=True,
    ):
        captions = captions_by_image.get(image_path.name, [])
        score = score_image(image_path)
        scored_records.append(
            ImageRecord(
                source="flickr8k_landscape_clip",
                image_path=image_path,
                caption=captions[0] if captions else None,
                category="landscape",
                score=score,
            )
        )

    scored_records.sort(key=lambda record: (record.score, record.image_path.name), reverse=True)
    selected = scored_records[:target_count]
    if len(selected) < target_count:
        raise ValueError(f"Only found {len(selected)} Flickr8k candidates, fewer than requested {target_count}")
    return selected


def select_ink_category_records(records: list[ImageRecord], category: str, target_count: int, seed: int) -> list[ImageRecord]:
    category_records = [record for record in records if record.category == category]
    if len(category_records) < target_count:
        raise ValueError(f"Only found {len(category_records)} records for ink category '{category}', fewer than requested {target_count}")

    shuffled = list(category_records)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:target_count], key=lambda record: record.image_path.name)


def safe_stem(text: str, fallback: str) -> str:
    base = text.strip() or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
    return (cleaned or fallback)[:80]


def open_record_image(record: ImageRecord) -> Image.Image:
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed")
    with Image.open(record.image_path) as img:
        return img.convert("RGB")


def save_record_image(record: ImageRecord, dst: Path, image_size: int) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if PIL_AVAILABLE:
            image = open_record_image(record)
            image = ImageOps.fit(image, (image_size, image_size), method=Image.BICUBIC)
            final_dst = dst.with_suffix(".png")
            image.save(final_dst)
            return final_dst
        final_dst = dst.with_suffix(record.image_path.suffix.lower())
        shutil.copy2(record.image_path, final_dst)
        return final_dst
    except Exception:
        return Path()


def export_domain(
    records: list[ImageRecord],
    out_dir: Path,
    image_size: int,
    seed: int,
    name_prefix: str,
) -> dict:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    counters = {"saved": 0, "failed": 0}
    manifest_rows: list[dict[str, str]] = []
    image_dir = out_dir

    for i, record in enumerate(
        tqdm(
            shuffled,
            desc=f"Export {out_dir.name}",
            ncols=PROGRESS_NCOLS,
            dynamic_ncols=False,
            leave=True,
        )
    ):
        file_name = f"{safe_stem(record.image_path.stem, name_prefix)}_{i:06d}"
        dst = image_dir / file_name
        saved_path = save_record_image(record, dst, image_size=image_size)
        if saved_path:
            counters["saved"] += 1
            manifest_rows.append(
                {
                    "file": str(saved_path),
                    "source": str(record.image_path),
                    "caption": record.caption or "",
                    "category": record.category or "",
                    "score": f"{record.score:.6f}",
                }
            )
        else:
            counters["failed"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "caption", "category", "score"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "total_input": len(shuffled),
        "saved": counters["saved"],
        "failed": counters["failed"],
        "manifest": str(out_dir / "manifest.csv"),
    }


def summarize_records(name: str, records: list[ImageRecord]) -> dict:
    with_caption = sum(1 for record in records if bool(record.caption and record.caption.strip()))
    categories: dict[str, int] = {}
    for category in INK_CATEGORY_ORDER + ("landscape",):
        count = sum(1 for record in records if record.category == category)
        if count > 0:
            categories[category] = count
    return {
        "dataset": name,
        "records": len(records),
        "with_caption": with_caption,
        "categories": categories,
        "score_min": min((record.score for record in records), default=0.0),
        "score_max": max((record.score for record in records), default=0.0),
    }


def preview_text(record: ImageRecord) -> str:
    if record.category:
        return f"{record.category}: {record.image_path.name}"
    if record.caption:
        return textwrap.shorten(record.caption, width=42, placeholder="...")
    return record.image_path.name


def save_preview_grid(records: list[ImageRecord], out_path: Path, title: str, max_items: int = 12, tile_size: int = 192) -> int:
    if not PIL_AVAILABLE:
        return 0

    decoded: list[tuple[Image.Image, ImageRecord]] = []
    for record in records:
        try:
            decoded.append((open_record_image(record), record))
        except Exception:
            continue
        if len(decoded) >= max_items:
            break

    if not decoded:
        return 0

    cols = min(4, len(decoded))
    rows = math.ceil(len(decoded) / cols)
    title_h = 36
    caption_h = 32

    canvas = Image.new("RGB", (cols * tile_size, title_h + rows * (tile_size + caption_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), f"{title} | preview={len(decoded)}", fill=(0, 0, 0))

    for i, (image, record) in enumerate(decoded):
        x = (i % cols) * tile_size
        y = title_h + (i // cols) * (tile_size + caption_h)
        tile = ImageOps.fit(image, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(tile, (x, y))
        draw.text((x + 4, y + tile_size + 6), preview_text(record)[:42], fill=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return len(decoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a compact CLIP-filtered landscape and ink dataset")
    parser.add_argument("--flickr-root", type=Path, default=Path("flickr8k"))
    parser.add_argument("--ink-root", type=Path, default=Path("ink_src"))
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--preview-root", type=Path, default=Path("outputs/data_preview"))
    parser.add_argument("--photo-count", type=int, default=1000)
    parser.add_argument("--ink-count", type=int, default=1000)
    parser.add_argument("--ink-category", type=str, default="shui_mo")
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-items", type=int, default=12)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    all_ink_records = load_ink_records(args.ink_root)
    flickr_records = load_flickr8k_landscape_records_with_clip(
        args.flickr_root,
        target_count=args.photo_count,
        clip_model_name=args.clip_model_name,
    )
    ink_records = select_ink_category_records(all_ink_records, args.ink_category, args.ink_count, seed=args.seed)

    stats = {
        "selected_flickr_landscape": summarize_records("selected_flickr_landscape", flickr_records),
        "selected_ink": summarize_records(f"selected_ink_{args.ink_category}", ink_records),
        "selection": {
            "photo_count": len(flickr_records),
            "ink_count": len(ink_records),
            "ink_category": args.ink_category,
            "photo_selector": "clip",
            "clip_model_name": args.clip_model_name,
            "clip_positive_prompts": CLIP_POSITIVE_PROMPTS,
            "clip_negative_prompts": CLIP_NEGATIVE_PROMPTS,
        },
    }

    stats["previews"] = {
        "flickr_landscape": save_preview_grid(
            flickr_records,
            args.preview_root / "flickr_landscape_preview.png",
            "flickr_landscape",
            max_items=args.preview_items,
        ),
        f"ink_{args.ink_category}": save_preview_grid(
            ink_records,
            args.preview_root / f"ink_{args.ink_category}_preview.png",
            f"ink_{args.ink_category}",
            max_items=args.preview_items,
        ),
    }

    if not args.skip_export:
        stats["exports"] = {
            "landscape": export_domain(
                flickr_records,
                args.out_root / "landscape",
                image_size=args.image_size,
                seed=args.seed,
                name_prefix="landscape",
            ),
            "ink": export_domain(
                ink_records,
                args.out_root / "ink" / args.ink_category,
                image_size=args.image_size,
                seed=args.seed + 1,
                name_prefix=args.ink_category,
            ),
        }

    args.preview_root.mkdir(parents=True, exist_ok=True)
    stats_path = args.preview_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved previews to {args.preview_root}")


if __name__ == "__main__":
    main()
