"""Read-only equivalence audit for canonical and generated dataset layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .coco import CocoDetectionDataset, read_coco_detection, resolve_coco_image_path
from .coordinates import coco_xywh_to_yolo, format_yolo_box, yolo_to_coco_xywh
from .manifests import SplitManifest, read_manifest
from ..progress import progress


@dataclass(frozen=True)
class AuditDifference:
    image_id: str
    expected: int | str
    actual: int | str


@dataclass(frozen=True)
class MigrationAuditReport:
    split_counts: dict[str, int]
    missing_ids: tuple[str, ...]
    overlapping_ids: tuple[str, ...]
    annotation_count_mismatches: tuple[AuditDifference, ...]
    maximum_yolo_round_trip_error: float
    historical_yolo_differences: tuple[AuditDifference, ...] = ()


def _read_yolo(path: Path) -> list[tuple[float, float, float, float]]:
    values: list[tuple[float, float, float, float]] = []
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"invalid YOLO row in {path}")
        values.append(tuple(float(value) for value in fields[1:]))  # type: ignore[arg-type]
    return values


def audit_dataset_migration(
    source_root: Path,
    annotations_path: Path,
    manifest_path: Path | SplitManifest,
    generated_yolo_root: Path,
    historical_yolo_root: Path | None = None,
) -> MigrationAuditReport:
    source: CocoDetectionDataset = read_coco_detection(Path(annotations_path))
    manifest = read_manifest(Path(manifest_path)) if isinstance(manifest_path, (str, Path)) else manifest_path
    by_stem = {Path(image.file_name).stem: image for image in source.images.values()}
    annotations_by_id: dict[int, list] = {}
    for annotation in source.annotations:
        annotations_by_id.setdefault(annotation.image_id, []).append(annotation)
    split_counts = {"train": 0, "val": 0, "test": 0}
    seen: dict[str, str] = {}
    missing: list[str] = []
    overlaps: list[str] = []
    mismatches: list[AuditDifference] = []
    historical_differences: list[AuditDifference] = []
    maximum_error = 0.0
    for split, image_id in progress(
        manifest.rows,
        description="audit dataset migration",
        total=len(manifest.rows),
    ):
        if split not in split_counts:
            raise ValueError(f"unsupported manifest split: {split}")
        split_counts[split] += 1
        if image_id in seen:
            overlaps.append(image_id)
        seen[image_id] = split
        image = by_stem.get(Path(image_id).stem)
        if image is None or not resolve_coco_image_path(Path(source_root), image).is_file():
            missing.append(image_id)
            continue
        expected = annotations_by_id.get(image.id, [])
        generated = _read_yolo(Path(generated_yolo_root) / split / "labels" / f"{image_id}.txt")
        if len(expected) != len(generated):
            mismatches.append(AuditDifference(image_id, len(expected), len(generated)))
        for annotation, encoded in zip(expected, generated):
            restored = yolo_to_coco_xywh(encoded, image.width, image.height)
            maximum_error = max(maximum_error, *(abs(a - b) for a, b in zip(annotation.bbox, restored)))
        if historical_yolo_root is not None:
            old = _read_yolo(Path(historical_yolo_root) / split / "labels" / f"{image_id}.txt")
            if len(old) != len(generated) or any(abs(a - b) > 1e-5 for left, right in zip(old, generated) for a, b in zip(left, right)):
                historical_differences.append(AuditDifference(image_id, len(old), len(generated)))
    return MigrationAuditReport(
        split_counts=split_counts,
        missing_ids=tuple(sorted(set(missing))),
        overlapping_ids=tuple(sorted(set(overlaps))),
        annotation_count_mismatches=tuple(mismatches),
        maximum_yolo_round_trip_error=maximum_error,
        historical_yolo_differences=tuple(historical_differences),
    )
