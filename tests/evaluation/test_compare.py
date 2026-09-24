from pathlib import Path

import pytest

from cabbage_detection.evaluation.compare import compare_to_published
from cabbage_detection.evaluation.detection import DetectionReport


def test_compare_exact_detection_values(tmp_path: Path):
    published = tmp_path / "published.csv"
    published.write_text("model,map50,map50_95,source\nfaster_rcnn,0.956,0.669,published-paper\n", encoding="utf-8")
    report = compare_to_published(DetectionReport(0.956, 0.669), published, "faster_rcnn")
    assert report.absolute_difference == {"map50": 0.0, "map50_95": 0.0}


def test_compare_requires_matching_metrics_and_preserves_source(tmp_path: Path):
    published = tmp_path / "published.csv"
    original = "model,map50,source\nfaster_rcnn,0.956,published-paper\n"
    published.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="published metrics missing"):
        compare_to_published(DetectionReport(0.956, 0.669), published, "faster_rcnn")
    assert published.read_text(encoding="utf-8") == original


def test_compare_rejects_unknown_model(tmp_path: Path):
    published = tmp_path / "published.csv"
    published.write_text("model,map50,map50_95,source\nfaster_rcnn,0.956,0.669,published-paper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not contain model"):
        compare_to_published(DetectionReport(0.956, 0.669), published, "ssd")
