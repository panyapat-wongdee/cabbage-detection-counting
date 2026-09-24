import pytest

from cabbage_detection.evaluation.records import Box, Detection
from cabbage_detection.metrics.boxes import box_iou


def test_box_iou_returns_overlap_ratio():
    assert box_iou((0, 0, 2, 2), (1, 1, 3, 3)) == pytest.approx(1 / 7)


def test_box_iou_rejects_inverted_box():
    with pytest.raises(ValueError, match="coordinates"):
        box_iou((2, 0, 1, 1), (0, 0, 1, 1))


def test_detection_and_box_are_available_records():
    target = Box((0, 0, 1, 1))
    detection = Detection(box=(0, 0, 1, 1), score=0.9)
    assert target.coordinates == detection.box
