from __future__ import annotations

import pytest

from cabbage_detection.evaluation.coco_detection import evaluate_coco_detection
from cabbage_detection.evaluation.records import Box, Detection, PredictionRecord, TargetRecord


def _records(*, false_positive: bool = False):
    detections = (Detection((0, 0, 10, 10), 0.9),)
    if false_positive:
        detections = (Detection((20, 20, 30, 30), 0.99),) + detections
    return (
        (PredictionRecord("a", detections), PredictionRecord("b", ())),
        (TargetRecord("a", (Box((0, 0, 10, 10)),)), TargetRecord("b", ())),
    )


def test_coco_perfect_prediction_has_perfect_ap() -> None:
    predictions, targets = _records()
    report = evaluate_coco_detection(predictions, targets, max_detections=100)
    assert report.backend == "pycocotools.coco_eval"
    assert report.map50 == pytest.approx(1.0)
    assert report.map50_95 == pytest.approx(1.0)


def test_coco_false_positive_lowers_ap() -> None:
    predictions, targets = _records(false_positive=True)
    report = evaluate_coco_detection(predictions, targets, max_detections=100)
    assert 0.0 <= report.map50 < 1.0
    assert 0.0 <= report.map50_95 < 1.0


def test_coco_empty_predictions_return_zero() -> None:
    predictions, targets = _records()
    report = evaluate_coco_detection(
        tuple(PredictionRecord(record.image_id, ()) for record in predictions),
        targets,
        max_detections=100,
    )
    assert report.map50 == 0.0
    assert report.map50_95 == 0.0


@pytest.mark.parametrize(
    "predictions, targets, message",
    [
        (
            (PredictionRecord("a", (Detection((0, 0, 10, 10), 0.9),)), PredictionRecord("a", ())),
            (TargetRecord("a", (Box((0, 0, 10, 10)),)),),
            "duplicate",
        ),
        (
            (PredictionRecord("missing", (Detection((0, 0, 10, 10), 0.9),)),),
            (TargetRecord("a", (Box((0, 0, 10, 10)),)),),
            "unknown",
        ),
        (
            (PredictionRecord("a", (Detection((0, 10, 10, 0), 0.9),)),),
            (TargetRecord("a", (Box((0, 0, 10, 10)),)),),
            "box",
        ),
        (
            (PredictionRecord("a", (Detection((0, 0, 10, 10), 0.9, class_id=2),)),),
            (TargetRecord("a", (Box((0, 0, 10, 10)),)),),
            "class",
        ),
    ],
)
def test_coco_rejects_invalid_records(predictions, targets, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_coco_detection(predictions, targets, max_detections=100)
