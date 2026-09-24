import pytest

from cabbage_detection.evaluation.records import Box, Detection
from cabbage_detection.metrics.counting import count_metrics, match_detections, match_pairs


def test_matching_is_one_to_one_and_counts_duplicates_as_false_positive():
    targets = [Box((0, 0, 2, 2))]
    predictions = [
        Detection((0, 0, 2, 2), 0.9),
        Detection((0, 0, 2, 2), 0.8),
    ]

    summary = match_detections(predictions, targets, iou_threshold=0.5)

    assert (summary.true_positives, summary.false_positives, summary.false_negatives) == (1, 1, 0)


def test_count_metrics_calculates_published_definitions():
    summary = match_detections(
        [Detection((0, 0, 1, 1), 0.9), Detection((4, 4, 5, 5), 0.9)],
        [Box((0, 0, 1, 1)), Box((2, 2, 3, 3)), Box((6, 6, 7, 7))],
        iou_threshold=0.5,
    )

    metrics = count_metrics(summary)

    assert metrics.predicted_count == 2
    assert metrics.actual_count == 3
    assert metrics.count_error == pytest.approx(-1 / 3)
    assert metrics.fdr == pytest.approx(0.5)
    assert metrics.fnr == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(0.4)


def test_matching_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([], [], iou_threshold=1.1)


def test_match_pairs_prefers_the_highest_iou_and_uses_each_box_once():
    targets = [Box((0, 0, 10, 10)), Box((20, 20, 30, 30))]
    predictions = [
        Detection((1, 1, 10, 10), 0.6),  # IoU 0.81 with target 0
        Detection((0, 0, 10, 10), 0.9),  # IoU 1.0 with target 0: wins it
        Detection((50, 50, 60, 60), 0.9),  # matches nothing
    ]

    pairs = match_pairs(predictions, targets, iou_threshold=0.5)

    assert pairs == [(1, 0)]
    summary = match_detections(predictions, targets, iou_threshold=0.5)
    assert summary.true_positives == len(pairs)
