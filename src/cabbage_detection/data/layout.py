"""Validation of prepared detection dataset layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    splits: tuple[str, ...]

    def image_ids(self, split: str) -> tuple[str, ...]:
        image_dir = self.root / split / "images"
        return tuple(sorted(path.stem for path in image_dir.iterdir() if path.is_file()))


def validate_dataset_root(root: Path) -> DatasetLayout:
    """Validate train/val/test image-label directories and matching stems."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist: {root}")
    splits = ("train", "val", "test")
    for split in splits:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"missing images or labels directory for split {split}")
        images = {path.stem for path in image_dir.iterdir() if path.is_file()}
        labels = {path.stem for path in label_dir.iterdir() if path.is_file()}
        missing_labels = images - labels
        orphan_labels = labels - images
        if missing_labels:
            raise ValueError(f"missing label for image(s): {', '.join(sorted(missing_labels))}")
        if orphan_labels:
            raise ValueError(f"label without image: {', '.join(sorted(orphan_labels))}")
        if not images:
            raise ValueError(f"split {split} contains no images")
    return DatasetLayout(root=root, splits=splits)
