"""COCOeval-backed detection metrics for historical-method profiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .detection import DetectionReport
from .records import PredictionRecord, TargetRecord


def _coco_modules() -> tuple[Any, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("install pycocotools for COCO-compatible evaluation") from error
    return COCO, COCOeval


def _validate_unique(records: Sequence[PredictionRecord] | Sequence[TargetRecord], label: str) -> dict[str, object]:
    by_id: dict[str, object] = {}
    for record in records:
        if record.image_id in by_id:
            raise ValueError(f"duplicate {label} image_id: {record.image_id}")
        by_id[record.image_id] = record
    return by_id


def evaluate_coco_detection(
    predictions: Sequence[PredictionRecord],
    targets: Sequence[TargetRecord],
    *,
    max_detections: int = 100,
) -> DetectionReport:
    """Evaluate canonical records with the torchvision notebook's COCOeval path."""

    if not isinstance(max_detections, int) or max_detections <= 0:
        raise ValueError("max_detections must be a positive integer")
    prediction_by_id = _validate_unique(predictions, "prediction")
    target_by_id = _validate_unique(targets, "target")
    unknown = set(prediction_by_id) - set(target_by_id)
    if unknown:
        raise ValueError(f"unknown prediction image_id: {sorted(unknown)[0]}")

    image_ids = sorted(target_by_id)
    numeric_ids = {image_id: index + 1 for index, image_id in enumerate(image_ids)}
    images: list[dict[str, object]] = [{"id": numeric_ids[image_id]} for image_id in image_ids]
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id in image_ids:
        target = target_by_id[image_id]
        assert isinstance(target, TargetRecord)
        for box in target.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.coordinates)
            if x2 <= x1 or y2 <= y1 or min(x1, y1) < 0:
                raise ValueError(f"invalid target box for image_id: {image_id}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": numeric_ids[image_id],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    COCO, COCOeval = _coco_modules()
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Cabbage"}],
    }
    coco_gt.createIndex()
    results: list[dict[str, object]] = []
    for image_id, record in prediction_by_id.items():
        assert isinstance(record, PredictionRecord)
        for detection in record.detections:
            if detection.class_id != 1:
                raise ValueError(f"unsupported prediction class_id: {detection.class_id}")
            x1, y1, x2, y2 = (float(value) for value in detection.box)
            if x2 <= x1 or y2 <= y1 or min(x1, y1) < 0:
                raise ValueError(f"invalid prediction box for image_id: {image_id}")
            results.append(
                {
                    "image_id": numeric_ids[image_id],
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(detection.score),
                }
            )
    if not annotations:
        return DetectionReport(map50=0.0, map50_95=0.0, backend="pycocotools.coco_eval")
    coco_dt = coco_gt.loadRes(results) if results else COCO()
    if not results:
        coco_dt.dataset = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "Cabbage"}]}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [numeric_ids[image_id] for image_id in image_ids]
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, max_detections]
    evaluator.evaluate()
    evaluator.accumulate()
    # ``COCOeval.summarize`` assumes the third maxDets entry is 100 and emits
    # ``-1`` when a caller intentionally configures a smaller limit. Read the
    # precision tensor at the configured limit instead.
    precision = evaluator.eval["precision"]

    def mean_valid(values: Any) -> float:
        flattened = [float(value) for value in values.reshape(-1).tolist() if float(value) >= 0]
        return sum(flattened) / len(flattened) if flattened else 0.0

    map50_95 = mean_valid(precision[:, :, 0, 0, -1])
    map50 = mean_valid(precision[0:1, :, 0, 0, -1])
    return DetectionReport(
        map50=map50,
        map50_95=map50_95,
        backend="pycocotools.coco_eval",
    )
