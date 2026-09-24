"""Strict validation for the immutable official dataset and split manifest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .coco import read_coco_detection, resolve_coco_image_path
from .manifests import SplitManifest
from ..progress import progress


@dataclass(frozen=True)
class DatasetValidationReport:
    image_count: int
    annotation_count: int
    split_counts: dict[str, int]
    annotations_sha256: str
    manifest_digest: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_dataset(images_root: Path, annotations_path: Path, manifest: SplitManifest) -> DatasetValidationReport:
    source = read_coco_detection(Path(annotations_path))
    by_stem = {Path(image.file_name).stem: image for image in source.images.values()}
    if len(by_stem) != len(source.images):
        raise ValueError("COCO image filenames must have unique stems")
    seen: dict[str, str] = {}
    missing: list[str] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for split, image_id in progress(
        manifest.rows,
        description="validate dataset",
        total=len(manifest.rows),
    ):
        if split not in split_counts:
            raise ValueError(f"unsupported manifest split: {split}")
        if image_id in seen:
            raise ValueError(f"image {image_id!r} appears in multiple splits")
        seen[image_id] = split
        split_counts[split] += 1
        image = by_stem.get(Path(image_id).stem)
        if image is None or not resolve_coco_image_path(Path(images_root), image).is_file():
            missing.append(f"{split}:{image_id}")
    if missing:
        raise ValueError("manifest images missing from official dataset: " + ", ".join(missing))
    if any(value == 0 for value in split_counts.values()):
        raise ValueError("manifest must contain train, val, and test rows")
    return DatasetValidationReport(
        image_count=len(source.images),
        annotation_count=len(source.annotations),
        split_counts=split_counts,
        annotations_sha256=sha256_file(Path(annotations_path)),
        manifest_digest=manifest.digest,
    )
