"""Manifest-backed torchvision dataset boundary with lazy heavyweight imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .coco import (
    CocoAnnotation,
    CocoDetectionDataset,
    CocoImage,
    coco_xywh_to_xyxy,
    read_coco_detection,
    resolve_coco_image_path,
)


class _RamImageCache:
    """Per-dataset RAM cache for decoded RGB images.

    Raw images are cached before transforms so stochastic training transforms
    are still sampled on every access. Each DataLoader worker owns its cache;
    this avoids sharing mutable image arrays between worker processes.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._images: dict[Path, Any] = {}

    @property
    def cached_image_count(self) -> int:
        return len(self._images)

    def get(self, path: Path) -> Any:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "install numpy and Pillow to load the torchvision dataset"
            ) from error
        if self.enabled and path in self._images:
            return self._images[path].copy()
        image = np.array(Image.open(path).convert("RGB"), copy=True)
        if self.enabled:
            self._images[path] = image
        return image.copy()


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    image: CocoImage
    path: Path
    annotations: tuple[CocoAnnotation, ...]


class CabbageDetectionDataset:
    """A COCO detection dataset restricted to an explicit manifest order."""

    def __init__(
        self,
        images_root: Path,
        annotations: Path,
        image_ids: Sequence[str],
        transforms: Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]] | None = None,
        cache: bool = False,
    ) -> None:
        self.images_root = Path(images_root)
        self.transforms = transforms
        self._ram_cache = _RamImageCache(cache)
        source: CocoDetectionDataset = read_coco_detection(Path(annotations))
        by_stem: dict[str, CocoImage] = {}
        for image in source.images.values():
            stem = Path(image.file_name).stem
            if stem in by_stem:
                raise ValueError(f"duplicate COCO image stem: {stem}")
            by_stem[stem] = image
        annotations_by_image: dict[int, list[CocoAnnotation]] = {}
        for annotation in source.annotations:
            annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
        samples: list[ImageSample] = []
        for image_id in image_ids:
            text_id = str(image_id)
            image = by_stem.get(Path(text_id).stem)
            if image is None:
                raise ValueError(f"manifest image_id {text_id!r} missing from COCO image table")
            image_path = resolve_coco_image_path(self.images_root, image)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            samples.append(
                ImageSample(text_id, image, image_path, tuple(annotations_by_image.get(image.id, ())))
            )
        self.samples = tuple(samples)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sample.image_id for sample in self.samples)

    @property
    def cache_enabled(self) -> bool:
        """Whether decoded images are retained in this dataset's RAM cache."""
        return self._ram_cache.enabled

    @property
    def cached_image_count(self) -> int:
        """Number of decoded images currently retained in RAM."""
        return self._ram_cache.cached_image_count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "install torch, numpy, and Pillow to load the torchvision dataset"
            ) from error
        sample = self.samples[index]
        image = self._ram_cache.get(sample.path)
        boxes = [coco_xywh_to_xyxy(annotation.bbox) for annotation in sample.annotations]
        labels = [1 for _ in boxes]
        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": sample.image.id,
            "image_id_text": sample.image_id,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous().float()
        if getattr(image, "dtype", None) is not None and image.dtype.kind in "ui":
            image_tensor = image_tensor / 255.0
        box_tensor = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(target["labels"], dtype=torch.int64)
        target.update(
            {
                "boxes": box_tensor,
                "labels": label_tensor,
                "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
                "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
            }
        )
        return image_tensor, target


def read_pascal_labels(path: Path) -> tuple[tuple[float, float, float, float], ...]:
    """Read the recovered Pascal-style ``class x_min y_min x_max y_max`` labels."""
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Pascal label row {line_number} must contain five values")
        try:
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"invalid Pascal label row {line_number}") from error
        if class_id != 0:
            raise ValueError(f"Pascal label row {line_number} has unsupported class id {class_id}")
        x_min, y_min, x_max, y_max = coordinates
        if min(coordinates) < 0:
            raise ValueError(f"Pascal label row {line_number} coordinates must be non-negative")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Pascal label row {line_number} coordinates must be ordered")
        boxes.append((x_min, y_min, x_max, y_max))
    return tuple(boxes)


class PascalDetectionDataset:
    """Manifest-ordered dataset for the recovered Pascal TXT export."""

    def __init__(
        self,
        images_root: Path,
        labels_root: Path,
        image_ids: Sequence[str],
        transforms: Callable[[Any, dict[str, Any]], tuple[Any, dict[str, Any]]] | None = None,
        cache: bool = False,
    ) -> None:
        self.images_root = Path(images_root)
        self.labels_root = Path(labels_root)
        self.transforms = transforms
        self._ram_cache = _RamImageCache(cache)
        samples: list[tuple[str, Path, tuple[tuple[float, float, float, float], ...]]] = []
        for image_id in image_ids:
            image_path = next((candidate for candidate in self.images_root.glob(f"{Path(image_id).stem}.*") if candidate.is_file()), None)
            if image_path is None:
                raise FileNotFoundError(self.images_root / f"{Path(image_id).stem}.*")
            label_path = self.labels_root / f"{Path(image_id).stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            samples.append((str(image_id), image_path, read_pascal_labels(label_path)))
        self.samples = tuple(samples)

    @property
    def image_ids(self) -> tuple[str, ...]:
        return tuple(sample[0] for sample in self.samples)

    @property
    def cache_enabled(self) -> bool:
        """Whether decoded images are retained in this dataset's RAM cache."""
        return self._ram_cache.enabled

    @property
    def cached_image_count(self) -> int:
        """Number of decoded images currently retained in RAM."""
        return self._ram_cache.cached_image_count

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("install torch, numpy, and Pillow to load the torchvision dataset") from error
        image_id, image_path, boxes = self.samples[index]
        image = self._ram_cache.get(image_path)
        target: dict[str, Any] = {
            "boxes": list(boxes),
            "labels": [1 for _ in boxes],
            "image_id_text": image_id,
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        image_tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous().float()
        if getattr(image, "dtype", None) is not None and image.dtype.kind in "ui":
            image_tensor = image_tensor / 255.0
        box_tensor = torch.as_tensor(target["boxes"], dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.as_tensor(target["labels"], dtype=torch.int64)
        target.update(
            {
                "boxes": box_tensor,
                "labels": label_tensor,
                "image_id": image_id,
                "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
                "iscrowd": torch.zeros((len(box_tensor),), dtype=torch.int64),
            }
        )
        return image_tensor, target


def collate_detection_batch(batch):
    """Keep variable-length detection targets as lists for torchvision."""
    images, targets = zip(*batch)
    return list(images), list(targets)
