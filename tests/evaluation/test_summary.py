"""Cross-split summary written at the evaluation root."""

from __future__ import annotations

import json
from pathlib import Path

from cabbage_detection.evaluation.summary import (
    collect_summary,
    render_summary_markdown,
    write_summary,
)


def _detection(path: Path, map50: float, images: int, backend: str = "cabbage_detection.ap_101"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": "detection",
                "scope": {"image_count": images},
                "metrics": {"map50": map50, "map50_95": map50 / 2, "backend": backend},
            }
        ),
        encoding="utf-8",
    )


def _counting(path: Path, images: int, f1: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": "counting",
                "scope": {"image_count": images},
                "metrics": {
                    "actual_count": 10,
                    "predicted_count": 11,
                    "true_positives": 9,
                    "false_positives": 2,
                    "false_negatives": 1,
                    "count_error": 0.1,
                    "fdr": 0.18,
                    "fnr": 0.1,
                    "f1": f1,
                },
                "metrics_macro": {"count_error": 0.2, "fdr": 0.2, "fnr": 0.2, "f1": 0.8},
            }
        ),
        encoding="utf-8",
    )


def test_collect_summary_gathers_every_split(tmp_path: Path):
    _detection(tmp_path / "train" / "detection.json", 0.9, 320)
    _counting(tmp_path / "train" / "counting.json", 320, 0.91)
    _detection(tmp_path / "test" / "detection.json", 0.8, 92)
    _counting(tmp_path / "test" / "counting.json", 92, 0.81)

    summary = collect_summary(tmp_path, ("train", "val", "test"))

    assert list(summary["splits"]) == ["train", "test"]
    assert summary["splits"]["train"]["detection"]["map50"] == 0.9
    assert summary["splits"]["test"]["counting"]["f1"] == 0.81
    assert summary["splits"]["test"]["counting"]["macro"]["f1"] == 0.8
    assert summary["detection_backends"] == ["cabbage_detection.ap_101"]


def test_collect_summary_records_each_backend_in_use(tmp_path: Path):
    _detection(tmp_path / "train" / "detection.json", 0.9, 320)
    _detection(tmp_path / "test" / "detection.json", 0.8, 92, backend="pycocotools.coco_eval")

    summary = collect_summary(tmp_path, ("train", "test"))

    assert summary["detection_backends"] == ["cabbage_detection.ap_101", "pycocotools.coco_eval"]


def test_collect_summary_handles_a_detection_only_evaluation(tmp_path: Path):
    _detection(tmp_path / "test" / "detection.json", 0.8, 92)

    summary = collect_summary(tmp_path, ("test",))

    assert "counting" not in summary["splits"]["test"]
    assert "detection" in summary["splits"]["test"]


def test_write_summary_produces_both_files(tmp_path: Path):
    _detection(tmp_path / "test" / "detection.json", 0.8, 92)
    _counting(tmp_path / "test" / "counting.json", 92, 0.81)

    json_path, markdown_path = write_summary(tmp_path, ("test",), "yolov8n evaluation")

    assert json_path.name == "summary.json"
    assert markdown_path.name == "summary.md"
    text = markdown_path.read_text(encoding="utf-8")
    assert "# yolov8n evaluation" in text
    assert "## Detection" in text
    assert "## Counting" in text
    assert "0.8000" in text


def test_markdown_states_when_nothing_was_written(tmp_path: Path):
    summary = collect_summary(tmp_path, ("test",))
    assert "No split reports" in render_summary_markdown(summary, "empty")
