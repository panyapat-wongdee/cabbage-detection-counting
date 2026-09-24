"""Materialization of YOLO labels from canonical COCO detection data."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .coco import CocoDetectionDataset, coco_xywh_to_xyxy, resolve_coco_image_path, xyxy_to_yolo
from .coordinates import format_yolo_box, yolo_to_coco_xywh
from .manifests import SplitManifest
from ..progress import progress


@dataclass(frozen=True)
class PreparationReport:
    image_count: int
    annotation_count: int
    output: Path
    source_manifest_digest: str
    split_counts: dict[str, int]
    maximum_round_trip_error: float
    link_mode: str


def _materialize(source: Path, destination: Path, link_mode: str) -> None:
    if link_mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError(f"cannot create hard link for {source}; use --link-mode copy explicitly") from error
    elif link_mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("link_mode must be hardlink or copy")


def prepare_yolo_dataset(
    source: CocoDetectionDataset,
    manifest: SplitManifest,
    output: Path,
    source_root: Path,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> PreparationReport:
    """Materialize normalized single-class labels and linked source images."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError("link_mode must be hardlink or copy")
    by_stem = {Path(image.file_name).stem: image for image in source.images.values()}
    annotations_by_image: dict[int, list] = {}
    for annotation in source.annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)
    seen: set[str] = set()
    split_counts = {"train": 0, "val": 0, "test": 0}
    selected_annotation_count = 0
    maximum_round_trip_error = 0.0
    for split, image_id in progress(
        manifest.rows,
        description="prepare YOLO dataset",
        total=len(manifest.rows),
    ):
        if split not in split_counts:
            raise ValueError(f"unsupported manifest split: {split}")
        if image_id in seen:
            raise ValueError(f"image {image_id!r} appears in multiple splits")
        seen.add(image_id)
        split_counts[split] += 1
        image = by_stem.get(image_id)
        if image is None:
            raise ValueError(f"manifest image_id not found in COCO images: {image_id}")
        source_path = resolve_coco_image_path(Path(source_root), image)
        if not source_path.is_file():
            raise ValueError(f"source image does not exist: {source_path}")
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        _materialize(source_path, image_dir / Path(image.file_name).name, link_mode)
        lines = []
        for annotation in annotations_by_image.get(image.id, []):
            selected_annotation_count += 1
            xyxy = coco_xywh_to_xyxy(annotation.bbox)
            encoded = xyxy_to_yolo(xyxy, image.width, image.height)
            serialized = tuple(float(value) for value in format_yolo_box(encoded).split())
            restored = yolo_to_coco_xywh(serialized, image.width, image.height)
            maximum_round_trip_error = max(maximum_round_trip_error, *(abs(a - b) for a, b in zip(annotation.bbox, restored)))
            lines.append("0 " + format_yolo_box(encoded))
        (label_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    dataset_root = output.resolve().as_posix()
    (output / "data.yaml").write_text(
        f'path: {json.dumps(dataset_root)}\n'
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: cabbage\n",
        encoding="utf-8",
    )
    report = PreparationReport(
        image_count=len(manifest.rows),
        annotation_count=selected_annotation_count,
        output=output,
        source_manifest_digest=manifest.digest,
        split_counts=split_counts,
        maximum_round_trip_error=maximum_round_trip_error,
        link_mode=link_mode,
    )
    (output / "preparation-report.json").write_text(json.dumps({
        "image_count": report.image_count,
        "annotation_count": report.annotation_count,
        "split_counts": report.split_counts,
        "source_manifest_digest": report.source_manifest_digest,
        "maximum_round_trip_error": report.maximum_round_trip_error,
        "link_mode": report.link_mode,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return report
