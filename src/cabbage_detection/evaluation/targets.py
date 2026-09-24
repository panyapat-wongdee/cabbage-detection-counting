"""Ground-truth target records built from the official COCO annotations.

Both frameworks are scored against the same annotation source, but their
prediction adapters do not share a coordinate space. The torchvision adapter
resizes each image with Albumentations and returns boxes in that resized space,
while the Ultralytics adapter returns boxes in original image pixels. Targets
are therefore mapped into the space of the framework that produced the
predictions instead of pretending the two pipelines are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..config import ExperimentConfig
from ..data.coco import CocoDetectionDataset, read_coco_detection, resolve_coco_image_path
from ..data.coordinates import coco_xywh_to_xyxy
from ..data.manifests import read_manifest
from ..data.official_layout import discover_official_dataset
from .records import Box, TargetRecord


@dataclass(frozen=True)
class EvaluationDataset:
    """One parsed official dataset, reused across every split of a run."""

    images_root: Path
    annotations_path: Path
    coco: CocoDetectionDataset
    images_by_stem: dict[str, Any]

    def image_path(self, image_id: str) -> Path:
        """Resolve the image file a manifest identifier names."""
        try:
            image = self.images_by_stem[Path(str(image_id)).stem]
        except KeyError as error:
            raise ValueError(f"manifest image_id {image_id!r} missing from COCO image table") from error
        return resolve_coco_image_path(self.images_root, image)


def load_evaluation_dataset(dataset_root: Path) -> EvaluationDataset:
    """Parse the official annotations once for a whole evaluation."""
    layout = discover_official_dataset(Path(dataset_root))
    coco = read_coco_detection(layout.annotations_path)
    return EvaluationDataset(
        images_root=layout.images_root,
        annotations_path=layout.annotations_path,
        coco=coco,
        images_by_stem=index_images_by_stem(coco),
    )


def split_image_ids(manifest_path: Path, split: str) -> tuple[str, ...]:
    """Return the manifest image identifiers for one split, in manifest order."""
    manifest = read_manifest(Path(manifest_path))
    selected = tuple(image_id for row_split, image_id in manifest.rows if row_split == split)
    if not selected:
        available = sorted({row_split for row_split, _ in manifest.rows})
        raise ValueError(f"split {split!r} is not present in the manifest; found {', '.join(available)}")
    return selected


def _scale_factors(config: ExperimentConfig, width: int, height: int) -> tuple[float, float]:
    """Return the x/y factors mapping original pixels into prediction space."""
    if config.framework != "torchvision":
        return 1.0, 1.0
    if width <= 0 or height <= 0:
        raise ValueError("COCO image dimensions must be positive to rescale targets")
    return config.image_size[1] / width, config.image_size[0] / height


def index_images_by_stem(source: CocoDetectionDataset) -> dict[str, Any]:
    """Map each COCO image to the manifest identifier that names it."""
    by_stem: dict[str, Any] = {}
    for image in source.images.values():
        stem = Path(image.file_name).stem
        if stem in by_stem:
            raise ValueError(f"duplicate COCO image stem: {stem}")
        by_stem[stem] = image
    return by_stem


def build_target_records(
    config: ExperimentConfig,
    annotations: Path | CocoDetectionDataset,
    image_ids: Sequence[str],
) -> tuple[TargetRecord, ...]:
    """Build canonical targets for ``image_ids`` in the framework's box space.

    A already-parsed dataset may be passed so one evaluation does not read the
    annotation file twice.
    """
    source = annotations if isinstance(annotations, CocoDetectionDataset) else read_coco_detection(Path(annotations))
    by_stem = index_images_by_stem(source)
    annotations_by_image: dict[int, list] = {}
    for annotation in source.annotations:
        annotations_by_image.setdefault(annotation.image_id, []).append(annotation)

    records: list[TargetRecord] = []
    for image_id in image_ids:
        text_id = str(image_id)
        image = by_stem.get(Path(text_id).stem)
        if image is None:
            raise ValueError(f"manifest image_id {text_id!r} missing from COCO image table")
        scale_x, scale_y = _scale_factors(config, image.width, image.height)
        boxes = []
        for annotation in annotations_by_image.get(image.id, ()):
            x1, y1, x2, y2 = coco_xywh_to_xyxy(annotation.bbox)
            boxes.append(Box((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)))
        records.append(TargetRecord(text_id, tuple(boxes)))
    return tuple(records)
