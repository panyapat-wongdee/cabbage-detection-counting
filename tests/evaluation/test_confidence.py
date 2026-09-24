"""Confidence widening for AP export and re-filtering for counting."""

from __future__ import annotations

from pathlib import Path

import pytest

from cabbage_detection.config import load_config
from cabbage_detection.evaluation.confidence import filter_by_confidence, lower_export_confidence
from cabbage_detection.evaluation.records import Detection, PredictionRecord


def _config():
    return load_config(Path("configs/torchvision/faster_rcnn.yaml"))


def test_lower_export_confidence_widens_both_thresholds():
    config = _config()
    assert config.confidence_threshold == 0.5
    lowered = lower_export_confidence(config, 0.001)
    assert lowered.confidence_threshold == 0.001
    assert lowered.final_postprocess.confidence_threshold == 0.001
    # The original stays untouched so counting can still use it.
    assert config.confidence_threshold == 0.5


def test_lower_export_confidence_refuses_to_raise_the_threshold():
    with pytest.raises(ValueError, match="must not exceed"):
        lower_export_confidence(_config(), 0.9)


def test_lower_export_confidence_rejects_a_value_outside_the_unit_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        lower_export_confidence(_config(), 1.5)


def test_filter_by_confidence_uses_a_strict_comparison():
    """``filter_torchvision_outputs`` drops ``score <= threshold``; match it."""
    records = [
        PredictionRecord(
            "image",
            (
                Detection((0, 0, 1, 1), 0.5),
                Detection((0, 0, 1, 1), 0.5001),
                Detection((0, 0, 1, 1), 0.9),
            ),
        )
    ]
    filtered = filter_by_confidence(records, 0.5)
    assert [detection.score for detection in filtered[0].detections] == [0.5001, 0.9]


def test_filter_by_confidence_keeps_every_image_including_emptied_ones():
    records = [
        PredictionRecord("a", (Detection((0, 0, 1, 1), 0.2),)),
        PredictionRecord("b", (Detection((0, 0, 1, 1), 0.8),)),
    ]
    filtered = filter_by_confidence(records, 0.5)
    assert [record.image_id for record in filtered] == ["a", "b"]
    assert filtered[0].detections == ()
    assert len(filtered[1].detections) == 1


def test_filter_by_confidence_is_a_no_op_on_already_filtered_records():
    records = [PredictionRecord("a", (Detection((0, 0, 1, 1), 0.7),))]
    assert filter_by_confidence(records, 0.5) == tuple(records)


def test_filter_by_confidence_rejects_an_out_of_range_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        filter_by_confidence([], 1.2)
