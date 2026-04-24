from __future__ import annotations

import csv
import io
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_COLUMNS = ("caption", "text", "prompt", "description", "title")
PATH_COLUMNS = ("image_path", "path", "file_path", "local_path", "image_file", "filepath")
BYTES_COLUMNS = ("image_bytes", "bytes", "jpg", "jpeg", "png", "webp", "image")
URL_COLUMNS = ("url", "image_url", "src", "source_url")


@dataclass
class ImageRecord:
    source: str
    image_path: Optional[Path] = None
    caption: Optional[str] = None
    image_bytes: Optional[bytes] = None
    url: Optional[str] = None



def _build_transform(image_size: int, augment: bool):
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if augment:
        ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return transforms.Compose(ops)



def collect_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in VALID_EXT])



def _detect_dataset_type(root: Path) -> str:
    if (root / "captions.txt").exists() and (root / "Images").exists():
        return "flickr8k"
    if list(root.glob("*.parquet")):
        return "laion-art"
    return "folder"



def _first_non_empty(row: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None



def _to_existing_path(value: object, root: Path) -> Optional[Path]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        raw = raw[7:]

    candidate = Path(raw)
    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
        joined = root / candidate
        if joined.exists():
            return joined
    return None



def _build_laion_record(row: dict, root: Path) -> Optional[ImageRecord]:
    caption = _first_non_empty(row, TEXT_COLUMNS)

    for key in PATH_COLUMNS:
        path = _to_existing_path(row.get(key), root)
        if path is not None:
            return ImageRecord(source="laion-art", image_path=path, caption=caption)

    for key in BYTES_COLUMNS:
        value = row.get(key)
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes) and len(value) > 0:
            return ImageRecord(source="laion-art", image_bytes=value, caption=caption)

    for key in URL_COLUMNS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return ImageRecord(source="laion-art", caption=caption, url=value.strip())

    if caption:
        return ImageRecord(source="laion-art", caption=caption)
    return None



def _load_flickr8k_records(root: Path, max_items: Optional[int] = None) -> list[ImageRecord]:
    image_root = root / "Images"
    caption_path = root / "captions.txt"

    if not image_root.exists():
        raise ValueError(f"Missing Flickr8k image folder: {image_root}")

    captions_by_image: dict[str, list[str]] = {}
    if caption_path.exists():
        with caption_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = (row.get("image") or "").strip()
                caption = (row.get("caption") or "").strip()
                if image_name and caption:
                    captions_by_image.setdefault(image_name, []).append(caption)

    records: list[ImageRecord] = []
    for image_path in collect_images(image_root):
        caption = None
        image_captions = captions_by_image.get(image_path.name)
        if image_captions:
            caption = image_captions[0]
        records.append(ImageRecord(source="flickr8k", image_path=image_path, caption=caption))
        if max_items is not None and len(records) >= max_items:
            break

    if not records:
        raise ValueError(f"No Flickr8k images found under {image_root}")
    return records



def _load_folder_records(root: Path, source: str = "folder", max_items: Optional[int] = None) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_path in collect_images(root):
        records.append(ImageRecord(source=source, image_path=image_path))
        if max_items is not None and len(records) >= max_items:
            break
    if not records:
        raise ValueError(f"No images found under {root}")
    return records



def _load_laion_records(root: Path, max_items: Optional[int] = None) -> list[ImageRecord]:
    parquet_files = sorted(root.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under {root}")

    records: list[ImageRecord] = []

    try:
        import pyarrow.parquet as pq  # type: ignore

        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(str(parquet_path))
            for batch in parquet_file.iter_batches(batch_size=1024):
                as_dict = batch.to_pydict()
                batch_len = len(next(iter(as_dict.values()))) if as_dict else 0
                for i in range(batch_len):
                    row = {k: as_dict[k][i] for k in as_dict}
                    rec = _build_laion_record(row, root)
                    if rec is not None:
                        records.append(rec)
                        if max_items is not None and len(records) >= max_items:
                            return records
    except ModuleNotFoundError:
        try:
            import pandas as pd  # type: ignore

            for parquet_path in parquet_files:
                df = pd.read_parquet(parquet_path)
                for _, row in df.iterrows():
                    rec = _build_laion_record(row.to_dict(), root)
                    if rec is not None:
                        records.append(rec)
                        if max_items is not None and len(records) >= max_items:
                            return records
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading laion-art parquet needs pyarrow or pandas. "
                "Install one of them before using parquet datasets."
            ) from exc

    if not records:
        raise ValueError(f"No usable LAION-art records found in {root}")
    return records



def load_image_records(root: str | Path, dataset_type: str = "auto", max_items: Optional[int] = None) -> list[ImageRecord]:
    root_path = Path(root)
    kind = dataset_type if dataset_type != "auto" else _detect_dataset_type(root_path)

    if kind == "flickr8k":
        return _load_flickr8k_records(root_path, max_items=max_items)
    if kind == "laion-art":
        return _load_laion_records(root_path, max_items=max_items)
    if kind == "ink_src":
        return _load_folder_records(root_path, source="ink_src", max_items=max_items)
    return _load_folder_records(root_path, source="folder", max_items=max_items)



def open_record_image(record: ImageRecord) -> Image.Image:
    if record.image_path is not None and record.image_path.exists():
        with Image.open(record.image_path) as img:
            return img.convert("RGB")
    if record.image_bytes is not None:
        with Image.open(io.BytesIO(record.image_bytes)) as img:
            return img.convert("RGB")
    raise ValueError("Record does not include local image data")


class FlexibleImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        augment: bool = False,
        dataset_type: str = "auto",
        max_items: Optional[int] = None,
    ):
        self.root = Path(root)
        self.records = load_image_records(self.root, dataset_type=dataset_type, max_items=max_items)
        self.transform = _build_transform(image_size=image_size, augment=augment)

        self.image_records = [
            rec
            for rec in self.records
            if rec.image_bytes is not None or (rec.image_path is not None and rec.image_path.exists())
        ]
        if not self.image_records:
            raise ValueError(f"No local images available under {self.root}")

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, idx: int):
        attempts = len(self.image_records)
        for offset in range(attempts):
            rec = self.image_records[(idx + offset) % attempts]
            try:
                image = open_record_image(rec)
                source_ref = str(rec.image_path) if rec.image_path is not None else (rec.url or rec.source)
                return self.transform(image), source_ref
            except Exception:
                continue
        raise RuntimeError(f"Unable to decode any image under {self.root}")


class ImageFolderDataset(FlexibleImageDataset):
    def __init__(self, root: str, image_size: int = 256, augment: bool = False):
        super().__init__(root=root, image_size=image_size, augment=augment, dataset_type="auto")


class MultiRootImageDataset(Dataset):
    def __init__(self, roots: Sequence[str], image_size: int = 256, augment: bool = False):
        self.datasets = [FlexibleImageDataset(root=r, image_size=image_size, augment=augment, dataset_type="auto") for r in roots]
        if not self.datasets:
            raise ValueError("At least one root is required")

        self.lengths = [len(ds) for ds in self.datasets]
        self.cumulative: list[int] = []
        total = 0
        for length in self.lengths:
            total += length
            self.cumulative.append(total)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        norm_idx = idx % self.total
        ds_i = bisect_right(self.cumulative, norm_idx)
        start = 0 if ds_i == 0 else self.cumulative[ds_i - 1]
        local_idx = norm_idx - start
        return self.datasets[ds_i][local_idx]


class UnpairedDataset(Dataset):
    def __init__(self, photo_root: str | Sequence[str], ink_root: str | Sequence[str], image_size: int = 256, augment: bool = False):
        photo_roots = [photo_root] if isinstance(photo_root, (str, Path)) else list(photo_root)
        ink_roots = [ink_root] if isinstance(ink_root, (str, Path)) else list(ink_root)

        self.photo_ds = self._build_domain_dataset(photo_roots, image_size=image_size, augment=augment)
        self.ink_ds = self._build_domain_dataset(ink_roots, image_size=image_size, augment=augment)
        self.length = max(len(self.photo_ds), len(self.ink_ds))

    @staticmethod
    def _build_domain_dataset(roots: Sequence[str | Path], image_size: int, augment: bool):
        normalized = [str(Path(r)) for r in roots]
        if len(normalized) == 1:
            return FlexibleImageDataset(normalized[0], image_size=image_size, augment=augment, dataset_type="auto")
        return MultiRootImageDataset(normalized, image_size=image_size, augment=augment)

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
