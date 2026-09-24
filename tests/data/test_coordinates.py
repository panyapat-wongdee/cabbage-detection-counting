import pytest

from cabbage_detection.data.coordinates import (
    assert_round_trip,
    coco_xywh_to_yolo,
    format_yolo_box,
    yolo_to_coco_xywh,
)


def test_yolo_round_trip_preserves_official_coco_box():
    original = (247, 390, 142, 96)
    encoded = coco_xywh_to_yolo(original, 515, 515)
    restored = yolo_to_coco_xywh(encoded, 515, 515)
    assert_round_trip(original, restored)


def test_serialized_yolo_box_round_trips_with_declared_tolerance():
    encoded = coco_xywh_to_yolo((1, 2, 3, 4), 515, 515)
    parsed = tuple(map(float, format_yolo_box(encoded).split()))
    assert yolo_to_coco_xywh(parsed, 515, 515) == pytest.approx((1, 2, 3, 4), abs=1e-6)
