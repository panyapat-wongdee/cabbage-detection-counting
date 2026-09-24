"""Read-only comparison of COCO and recovered Pascal training inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .coco import coco_xywh_to_xyxy, read_coco_detection, resolve_coco_image_path
from .manifests import read_manifest
from .official_layout import discover_official_dataset
from .torchvision_dataset import read_pascal_labels
from ..progress import progress


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_historical_inputs(coco_root: Path, pascal_root: Path, manifest: Path) -> dict[str, Any]:
    """Compare every manifest sample without modifying either dataset tree."""

    layout = discover_official_dataset(Path(coco_root))
    coco = read_coco_detection(layout.annotations_path)
    manifest_data = read_manifest(Path(manifest))
    by_stem = {Path(image.file_name).stem: image for image in coco.images.values()}
    annotations_by_image: dict[int, list[tuple[float, float, float, float]]] = {}
    category_ids_by_image: dict[int, set[int]] = {}
    for annotation in coco.annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(coco_xywh_to_xyxy(annotation.bbox))
        category_ids_by_image.setdefault(annotation.image_id, set()).add(annotation.category_id)
    split_counts = {split: 0 for split in ("train", "val", "test")}
    empty_coco = 0
    empty_pascal = 0
    mismatches: list[dict[str, Any]] = []
    checked = 0
    for split, image_id in progress(
        manifest_data.rows,
        description="audit historical inputs",
        total=len(manifest_data.rows),
    ):
        split_counts[split] = split_counts.get(split, 0) + 1
        text_id = str(image_id)
        image = by_stem.get(Path(text_id).stem)
        pascal_image = next(
            (candidate for candidate in (Path(pascal_root) / split / "images").glob(f"{Path(text_id).stem}.*") if candidate.is_file()),
            None,
        )
        pascal_label = Path(pascal_root) / split / "labels" / f"{Path(text_id).stem}.txt"
        reasons: list[str] = []
        coco_boxes: list[tuple[float, float, float, float]] = []
        if image is None:
            reasons.append("missing_coco_image")
        else:
            coco_boxes = annotations_by_image.get(image.id, [])
            if any(category_id != 1 for category_id in category_ids_by_image.get(image.id, set())):
                reasons.append("category_id_mismatch")
        if pascal_image is None:
            reasons.append("missing_pascal_image")
        if not pascal_label.is_file():
            reasons.append("missing_pascal_label")
            pascal_boxes: tuple[tuple[float, float, float, float], ...] = ()
        else:
            pascal_boxes = read_pascal_labels(pascal_label)
        if not coco_boxes:
            empty_coco += 1
        if not pascal_boxes:
            empty_pascal += 1
        if image is not None and pascal_image is not None:
            coco_image_path = resolve_coco_image_path(layout.images_root, image)
            if _sha256(coco_image_path) != _sha256(pascal_image):
                reasons.append("image_digest_mismatch")
        if len(coco_boxes) != len(pascal_boxes):
            reasons.append("annotation_count_mismatch")
        elif any(
            abs(left - right) > 1e-4
            for left_box, right_box in zip(sorted(coco_boxes), sorted(pascal_boxes))
            for left, right in zip(left_box, right_box)
        ):
            reasons.append("box_coordinates_mismatch")
        checked += 1
        if reasons:
            mismatches.append({"split": split, "image_id": text_id, "reasons": reasons})
    return {
        "manifest_digest": manifest_data.digest,
        "checked_images": checked,
        "split_counts": split_counts,
        "empty_images": {"coco": empty_coco, "pascal": empty_pascal},
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "equivalent": not mismatches,
    }
