"""Canonical conversion for native torchvision prediction dictionaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..evaluation.records import Detection, PostprocessingMetadata, PredictionRecord


def _values(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def convert_torchvision_predictions(
    image_ids: Sequence[str],
    outputs: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
    nms_iou_threshold: float = 0.5,
) -> list[PredictionRecord]:
    """Convert already-NMS'd torchvision outputs and filter confidence once."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not len(image_ids) == len(outputs):
        raise ValueError("image_ids and outputs must have equal lengths")
    metadata = PostprocessingMetadata(
        confidence_owner="repository",
        nms_owner="torchvision",
        confidence_threshold=float(confidence_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        native_nms_applied=True,
    )
    records: list[PredictionRecord] = []
    for image_id, output in zip(image_ids, outputs):
        missing = {"boxes", "scores", "labels"} - set(output)
        if missing:
            raise ValueError(f"torchvision output missing: {', '.join(sorted(missing))}")
        boxes = _values(output["boxes"])
        scores = _values(output["scores"])
        labels = _values(output["labels"])
        if not len(boxes) == len(scores) == len(labels):
            raise ValueError("torchvision boxes, scores, and labels must have equal lengths")
        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            score_value = float(score)
            if not math.isfinite(score_value):
                raise ValueError("torchvision scores must be finite")
            if score_value < confidence_threshold:
                continue
            if len(box) != 4:
                raise ValueError("torchvision boxes must contain four coordinates")
            coordinates = tuple(float(value) for value in box)
            if coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]:
                raise ValueError("torchvision boxes must be ordered xyxy")
            detections.append(Detection(coordinates, score_value, int(label)))
        records.append(PredictionRecord(str(image_id), tuple(detections), metadata))
    return records
