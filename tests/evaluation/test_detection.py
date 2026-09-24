import pytest

from cabbage_detection.evaluation.detection import evaluate_detection
from cabbage_detection.evaluation.records import Box, Detection, PredictionRecord, TargetRecord


def test_perfect_predictions_have_perfect_map():
    predictions = (PredictionRecord("a", (Detection((0, 0, 1, 1), 0.9),)),)
    targets = (TargetRecord("a", (Box((0, 0, 1, 1)),)),)

    report = evaluate_detection(predictions, targets)

    assert report.map50 == pytest.approx(1.0)
    assert report.map50_95 == pytest.approx(1.0)


def test_duplicate_predictions_reduce_map():
    predictions = (
        PredictionRecord("a", (Detection((2, 2, 3, 3), 0.9), Detection((0, 0, 1, 1), 0.8))),
    )
    targets = (TargetRecord("a", (Box((0, 0, 1, 1)),)),)

    report = evaluate_detection(predictions, targets)

    assert 0 < report.map50 < 1


def test_overlapping_targets_use_an_unmatched_target_for_each_prediction():
    predictions = (
        PredictionRecord(
            "a",
            (
                Detection((0, 0, 10, 10), 0.9),
                Detection((0, 0, 10, 10), 0.8),
            ),
        ),
    )
    targets = (
        TargetRecord(
            "a",
            (
                Box((0, 0, 10, 10)),
                Box((1, 0, 11, 10)),
            ),
        ),
    )

    report = evaluate_detection(predictions, targets)

    assert report.map50 == pytest.approx(1.0)


def test_tied_scores_in_one_image_are_ordered_without_comparing_detections():
    """Equal scores must not fall through to an unorderable Detection compare."""
    predictions = [
        PredictionRecord(
            "image",
            (
                Detection((0.0, 0.0, 10.0, 10.0), 0.5),
                Detection((20.0, 20.0, 30.0, 30.0), 0.5),
            ),
        )
    ]
    targets = [TargetRecord("image", (Box((0.0, 0.0, 10.0, 10.0)), Box((20.0, 20.0, 30.0, 30.0))))]

    report = evaluate_detection(predictions, targets)

    assert report.map50 == 1.0


def test_matching_is_recomputed_per_threshold_despite_shared_ious():
    """A pair that matches at 0.50 but not at higher IoU must not stay matched."""
    predictions = [PredictionRecord("image", (Detection((0.0, 0.0, 10.0, 6.0), 0.9),))]
    targets = [TargetRecord("image", (Box((0.0, 0.0, 10.0, 10.0)),))]

    report = evaluate_detection(predictions, targets)

    # IoU is exactly 0.6: counted at thresholds 0.50, 0.55 and 0.60, missed at
    # 0.65 upward, so three of the ten thresholds contribute.
    assert report.map50 == pytest.approx(1.0)
    assert report.map50_95 == pytest.approx(0.3)
