import pytest

from cabbage_detection.evaluation.counting import evaluate_counting
from cabbage_detection.evaluation.records import Box, Detection, PredictionRecord, TargetRecord


def test_evaluate_counting_joins_by_image_id_not_input_order():
    predictions = (
        PredictionRecord("b", (Detection((2, 2, 3, 3), 0.9),)),
        PredictionRecord("a", (Detection((0, 0, 1, 1), 0.9),)),
    )
    targets = (
        TargetRecord("a", (Box((0, 0, 1, 1)),)),
        TargetRecord("b", (Box((2, 2, 3, 3)),)),
    )

    report = evaluate_counting(predictions, targets, iou_threshold=0.5)

    assert report.totals.actual_count == 2
    assert report.totals.predicted_count == 2
    assert report.totals.f1 == 1.0
    assert report.match_totals is not None
    assert report.match_totals.true_positives == 2


def test_evaluate_counting_rejects_unknown_prediction_image():
    with pytest.raises(ValueError, match="unknown"):
        evaluate_counting(
            (PredictionRecord("unknown", ()),),
            (TargetRecord("known", ()),),
            iou_threshold=0.5,
        )
