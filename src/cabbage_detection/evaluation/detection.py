"""Framework-neutral mAP evaluation for canonical detection records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .records import PredictionRecord, TargetRecord
from ..metrics.boxes import box_iou
from ..progress import progress


@dataclass(frozen=True)
class DetectionReport:
    map50: float
    map50_95: float
    backend: str = "cabbage_detection.ap_101"


def _scored_detections(
    predictions: Sequence[PredictionRecord], targets: Sequence[TargetRecord]
) -> list[tuple[str, tuple[float, ...]]]:
    """Return per-detection IoUs against their image's targets, in score order.

    The IoU of a detection against a target does not depend on the AP
    threshold, so it is computed once and reused for all ten thresholds
    instead of being recomputed per threshold.
    """
    target_map = {record.image_id: record for record in targets}
    # Sort on an explicit key: score ties within one image would otherwise fall
    # through to comparing ``Detection`` objects, which are not orderable. The
    # sort is stable, so tied detections keep their record order.
    scored = sorted(
        (
            (detection.score, record.image_id, detection)
            for record in predictions
            for detection in record.detections
        ),
        key=lambda item: (-item[0], item[1]),
    )
    rows: list[tuple[str, tuple[float, ...]]] = []
    for _, image_id, detection in scored:
        target = target_map.get(image_id)
        boxes = target.boxes if target else ()
        rows.append(
            (image_id, tuple(box_iou(detection.box, box.coordinates) for box in boxes))
        )
    return rows


def _average_precision(
    rows: Sequence[tuple[str, tuple[float, ...]]], total_targets: int, threshold: float
) -> float:
    """Interpolate precision over 101 recall points at one IoU threshold."""
    if not total_targets:
        return 0.0
    matched: dict[str, set[int]] = {}
    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for image_id, ious in rows:
        taken = matched.setdefault(image_id, set())
        best_iou = 0.0
        best_index = -1
        for index, iou in enumerate(ious):
            if index not in taken and iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= threshold:
            taken.add(best_index)
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        recalls.append(cumulative_tp / total_targets)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
    if not recalls:
        return 0.0
    envelope = [
        max((precision for recall, precision in zip(recalls, precisions) if recall >= point), default=0.0)
        for point in [index / 100 for index in range(101)]
    ]
    return sum(envelope) / len(envelope)


def evaluate_detection(
    predictions: Sequence[PredictionRecord], targets: Sequence[TargetRecord]
) -> DetectionReport:
    """Calculate AP at IoU 0.50 and averaged over 0.50:0.95."""
    target_ids = {record.image_id for record in targets}
    prediction_ids = {record.image_id for record in predictions}
    if prediction_ids - target_ids:
        raise ValueError("predictions contain unknown image_id")
    rows = _scored_detections(predictions, targets)
    total_targets = sum(len(record.boxes) for record in targets)
    map50 = _average_precision(rows, total_targets, 0.5)
    thresholds = [0.5 + index * 0.05 for index in range(10)]
    map50_95 = sum(
        _average_precision(rows, total_targets, threshold)
        for threshold in progress(thresholds, description="evaluate mAP", total=len(thresholds))
    ) / len(thresholds)
    return DetectionReport(map50=map50, map50_95=map50_95)
