from __future__ import annotations

import argparse
import csv
import json
import math
import random
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageOps

from case9.data.dataset import ImageRecord, load_image_records, open_record_image


SPLITS = ("train", "val", "test")


def normalize_limit(value: int) -> Optional[int]:
    return None if value <= 0 else value


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_n = int(total * train_ratio)
    val_n = int(total * val_ratio)
    test_n = max(total - train_n - val_n, 0)
    return train_n, val_n, test_n


def pick_split(index: int, train_n: int, val_n: int) -> str:
    if index < train_n:
        return "train"
    if index < train_n + val_n:
        return "val"
    return "test"


def safe_stem(text: str, fallback: str) -> str:
    base = text.strip() or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in base)
    return (cleaned or fallback)[:80]


def save_record_image(record: ImageRecord, dst: Path, image_size: int) -> bool:
    try:
        image = open_record_image(record)
        image = ImageOps.fit(image, (image_size, image_size), method=Image.BICUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst)
        return True
    except Exception:
        return False


def source_string(record: ImageRecord) -> str:
    if record.image_path is not None:
        return str(record.image_path)
    if record.url:
        return record.url
    return record.source


def export_domain(
    records: list[ImageRecord],
    out_dir: Path,
    image_size: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    name_prefix: str,
) -> dict:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_n, val_n, _ = split_counts(len(shuffled), train_ratio, val_ratio)
    counters = {"train": 0, "val": 0, "test": 0, "failed": 0}
    manifest_rows: list[dict[str, str]] = []

    for i, record in enumerate(shuffled):
        split = pick_split(i, train_n, val_n)
        stem = record.image_path.stem if record.image_path is not None else f"{name_prefix}_{i:06d}"
        file_name = f"{safe_stem(stem, name_prefix)}_{i:06d}.png"
        dst = out_dir / split / file_name

        if save_record_image(record, dst, image_size=image_size):
            counters[split] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "file": str(dst),
                    "source": source_string(record),
                    "caption": record.caption or "",
                }
            )
        else:
            counters["failed"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "source", "caption"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "total_input": len(shuffled),
        "train": counters["train"],
        "val": counters["val"],
        "test": counters["test"],
        "failed": counters["failed"],
        "manifest": str(out_dir / "manifest.csv"),
    }


def summarize_records(name: str, records: list[ImageRecord]) -> dict:
    with_caption = sum(1 for r in records if bool(r.caption and r.caption.strip()))
    path_count = sum(1 for r in records if r.image_path is not None)
    bytes_count = sum(1 for r in records if r.image_bytes is not None)
    url_only = sum(1 for r in records if r.url is not None and r.image_path is None and r.image_bytes is None)
    local_ready = sum(
        1
        for r in records
        if r.image_bytes is not None or (r.image_path is not None and r.image_path.exists())
    )
    return {
        "dataset": name,
        "records": len(records),
        "with_caption": with_caption,
        "path_records": path_count,
        "bytes_records": bytes_count,
        "url_only_records": url_only,
        "local_image_ready": local_ready,
    }


def preview_text(record: ImageRecord) -> str:
    if record.caption:
        return textwrap.shorten(record.caption, width=42, placeholder="…")
    if record.image_path is not None:
        return record.image_path.name
    if record.url:
        return textwrap.shorten(record.url, width=42, placeholder="…")
    return record.source


def save_preview_grid(records: list[ImageRecord], out_path: Path, title: str, max_items: int = 12, tile_size: int = 192) -> int:
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
        draw.text((x + 4, y + tile_size + 6), preview_text(record), fill=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return len(decoded)


def load_records(dataset_root: Optional[Path], dataset_type: str, limit: Optional[int]) -> list[ImageRecord]:
    if dataset_root is None:
        return []
    if not dataset_root.exists():
        print(f"[skip] {dataset_type}: missing path {dataset_root}")
        return []
    return load_image_records(dataset_root, dataset_type=dataset_type, max_items=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and preview flickr8k / ink_src / laion-art datasets")
    parser.add_argument("--flickr-root", type=Path, default=Path("flickr8k"))
    parser.add_argument("--ink-root", type=Path, default=Path("ink_src"))
    parser.add_argument("--laion-root", type=Path, default=Path("laion-art"))
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--preview-root", type=Path, default=Path("outputs/data_preview"))
    parser.add_argument("--limit-flickr", type=int, default=0, help="0 means all")
    parser.add_argument("--limit-ink", type=int, default=0, help="0 means all")
    parser.add_argument("--limit-laion", type=int, default=20000, help="0 means all")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--preview-items", type=int, default=12)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    flickr_records = load_records(args.flickr_root, "flickr8k", normalize_limit(args.limit_flickr))
    ink_records = load_records(args.ink_root, "ink_src", normalize_limit(args.limit_ink))
    laion_records = load_records(args.laion_root, "laion-art", normalize_limit(args.limit_laion))

    photo_records = flickr_records + laion_records

    if not photo_records:
        raise ValueError("No photo-domain records loaded. Check flickr8k/laion-art paths and parquet dependencies.")
    if not ink_records:
        raise ValueError("No ink-domain records loaded. Check ink_src path.")

    stats = {
        "flickr8k": summarize_records("flickr8k", flickr_records),
        "ink_src": summarize_records("ink_src", ink_records),
        "laion-art": summarize_records("laion-art", laion_records),
        "photo_domain_total": len(photo_records),
        "ink_domain_total": len(ink_records),
    }

    previews = {
        "flickr8k": save_preview_grid(
            flickr_records,
            args.preview_root / "flickr8k_preview.png",
            "flickr8k",
            max_items=args.preview_items,
        )
        if flickr_records
        else 0,
        "ink_src": save_preview_grid(
            ink_records,
            args.preview_root / "ink_src_preview.png",
            "ink_src",
            max_items=args.preview_items,
        ),
        "laion-art": save_preview_grid(
            laion_records,
            args.preview_root / "laion_art_preview.png",
            "laion-art",
            max_items=args.preview_items,
        )
        if laion_records
        else 0,
        "photo_domain": save_preview_grid(
            photo_records,
            args.preview_root / "photo_domain_preview.png",
            "photo_domain(flickr8k+laion-art)",
            max_items=args.preview_items,
        ),
        "ink_domain": save_preview_grid(
            ink_records,
            args.preview_root / "ink_domain_preview.png",
            "ink_domain",
            max_items=args.preview_items,
        ),
    }

    stats["previews"] = previews

    if not args.skip_export:
        photo_export = export_domain(
            photo_records,
            args.out_root / "landscape",
            image_size=args.image_size,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            name_prefix="photo",
        )
        ink_export = export_domain(
            ink_records,
            args.out_root / "ink",
            image_size=args.image_size,
            seed=args.seed + 1,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            name_prefix="ink",
        )
        stats["exports"] = {
            "landscape": photo_export,
            "ink": ink_export,
        }

    args.preview_root.mkdir(parents=True, exist_ok=True)
    stats_path = args.preview_root / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"saved previews to {args.preview_root}")


if __name__ == "__main__":
    main()
