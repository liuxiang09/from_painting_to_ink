from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    source: str
    image_path: Path
    caption: Optional[str] = None


def _build_transform(image_size: int, augment: bool, color_jitter_strength: float = 0.05):
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_jitter_strength > 0:
            ops.append(
                transforms.ColorJitter(
                    brightness=color_jitter_strength,
                    contrast=color_jitter_strength,
                    saturation=color_jitter_strength,
                    hue=min(0.5, color_jitter_strength * 0.4),
                )
            )
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(ops)


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise ValueError(f"Missing directory: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory: {root}")
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])


def load_flickr8k_records(root: str | Path, max_items: Optional[int] = None) -> list[ImageRecord]:
    root_path = Path(root)
    image_root = root_path / "Images"
    caption_path = root_path / "captions.txt"

    if not image_root.exists():
        raise ValueError(f"Missing Flickr8k image folder: {image_root}")
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

    records: list[ImageRecord] = []
    for image_path in collect_images(image_root):
        image_captions = captions_by_image.get(image_path.name, [])
        caption = image_captions[0] if image_captions else None
        records.append(ImageRecord(source="flickr8k", image_path=image_path, caption=caption))
        if max_items is not None and len(records) >= max_items:
            break

    if not records:
        raise ValueError(f"No Flickr8k images found under {image_root}")
    return records


def load_flat_image_records(root: str | Path, source: str, max_items: Optional[int] = None) -> list[ImageRecord]:
    root_path = Path(root)
    records: list[ImageRecord] = []
    for image_path in collect_images(root_path):
        records.append(ImageRecord(source=source, image_path=image_path))
        if max_items is not None and len(records) >= max_items:
            break
    if not records:
        raise ValueError(f"No images found under {root_path}")
    return records


def open_record_image(record: ImageRecord) -> Image.Image:
    with Image.open(record.image_path) as img:
        return img.convert("RGB")


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 256, augment: bool = False, color_jitter_strength: float = 0.05):
        self.root = Path(root)
        self.image_paths = collect_images(self.root)
        self.transform = _build_transform(
            image_size=image_size,
            augment=augment,
            color_jitter_strength=color_jitter_strength,
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        return self.transform(image), str(image_path)


class UnpairedDataset(Dataset):
    def __init__(
        self,
        photo_root: str | Path,
        ink_root: str | Path,
        image_size: int = 256,
        augment: bool = False,
        photo_color_jitter_strength: float = 0.05,
        ink_color_jitter_strength: float = 0.05,
    ):
        self.photo_ds = ImageFolderDataset(
            photo_root,
            image_size=image_size,
            augment=augment,
            color_jitter_strength=photo_color_jitter_strength,
        )
        self.ink_ds = ImageFolderDataset(
            ink_root,
            image_size=image_size,
            augment=augment,
            color_jitter_strength=ink_color_jitter_strength,
        )
        self.length = max(len(self.photo_ds), len(self.ink_ds))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        photo_idx = idx % len(self.photo_ds)
        ink_idx = idx % len(self.ink_ds)

        photo_img, photo_path = self.photo_ds[photo_idx]
        ink_img, ink_path = self.ink_ds[ink_idx]
        return {
            "photo": photo_img,
            "ink": ink_img,
            "photo_path": photo_path,
            "ink_path": ink_path,
        }
