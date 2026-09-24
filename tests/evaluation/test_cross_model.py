"""Cross-model comparison table built from several evaluated run outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cabbage_detection.evaluation.cross_model import (
    collect_cross_model,
    discover_models,
    render_cross_model_markdown,
    write_cross_model,
)


def _write_model(
    root: Path,
    model: str,
    split: str = "test",
    *,
    map50: float = 0.9,
    backend: str = "cabbage_detection.ap_101",
    images: int = 92,
    counting_confidence: float = 0.5,
    deviations: tuple[str, ...] = (),
    with_counting: bool = True,
) -> Path:
    model_dir = root / model
    split_dir = model_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    detection = {
        "image_count": images,
        "backend": backend,
        "map50": map50,
        "map50_95": map50 / 2,
    }
    counting = {
        "image_count": images,
        "actual_count": 100,
        "predicted_count": 104,
        "true_positives": 96,
        "false_positives": 8,
        "false_negatives": 4,
        "count_error": 0.04,
        "fdr": 0.077,
        "fnr": 0.04,
        "f1": 0.94,
    }
    entry: dict[str, object] = {"detection": detection}
    if with_counting:
        entry["counting"] = {**counting, "macro": {"f1": 0.93}}
    (model_dir / "summary.json").write_text(
        json.dumps({"splits": {split: entry}}), encoding="utf-8"
    )
    (split_dir / "detection.json").write_text(
        json.dumps(
            {
                "metrics": detection,
                "evaluation": {
                    "ap_iou_threshold": 0.5,
                    "ap_iou_range": "0.50:0.95 step 0.05",
                    "backend": backend,
                    "export_confidence_threshold": 0.001,
                },
            }
        ),
        encoding="utf-8",
    )
    if with_counting:
        (split_dir / "counting.json").write_text(
            json.dumps(
                {
                    "metrics": counting,
                    "evaluation": {
                        "iou_threshold": 0.5,
                        "confidence_threshold": counting_confidence,
                        "matching": "one_to_one_greedy_highest_iou",
                        "aggregation": "micro",
                    },
                }
            ),
            encoding="utf-8",
        )
    (split_dir / "metadata.json").write_text(
        json.dumps(
            {
                "deviations": list(deviations),
                "git_revision": "a" * 40,
                "config": {"artifacts": {"result_status": "reproduction_candidate"}},
            }
        ),
        encoding="utf-8",
    )
    return model_dir


def test_discover_models_uses_reporting_order_and_ignores_other_directories(tmp_path: Path):
    _write_model(tmp_path, "yolov8n")
    _write_model(tmp_path, "faster_rcnn")
    _write_model(tmp_path, "ssdlite")
    (tmp_path / "yolov8n-native").mkdir()
    (tmp_path / "smoke").mkdir()

    assert discover_models(tmp_path) == ("faster_rcnn", "yolov8n", "ssdlite")


def test_collect_labels_framework_and_study_scope(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", map50=0.97)
    _write_model(tmp_path, "yolov8n", map50=0.96)
    _write_model(tmp_path, "ssdlite", map50=0.82)

    payload = collect_cross_model(tmp_path, "test")

    assert [record["model"] for record in payload["models"]] == [
        "faster_rcnn",
        "yolov8n",
        "ssdlite",
    ]
    scopes = {record["model"]: record["study_scope"] for record in payload["models"]}
    assert scopes["yolov8n"] == "paper_model"
    assert scopes["ssdlite"] == "supplementary_unreported"
    frameworks = {record["model"]: record["framework"] for record in payload["models"]}
    assert frameworks == {
        "faster_rcnn": "torchvision",
        "yolov8n": "ultralytics",
        "ssdlite": "torchvision",
    }
    assert payload["image_count"] == 92
    assert payload["settings"]["detection"]["backend"] == "cabbage_detection.ap_101"


def test_mixed_ap_backends_are_rejected(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", backend="pycocotools_coco_eval")
    _write_model(tmp_path, "yolov8n", backend="cabbage_detection.ap_101")

    with pytest.raises(ValueError, match="AP backend"):
        collect_cross_model(tmp_path, "test")


def test_mixed_counting_confidence_is_rejected(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", counting_confidence=0.5)
    _write_model(tmp_path, "yolov8n", counting_confidence=0.25)

    with pytest.raises(ValueError, match="counting confidence threshold"):
        collect_cross_model(tmp_path, "test")


def test_mixed_image_counts_are_rejected(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", images=92)
    _write_model(tmp_path, "yolov8n", images=91)

    with pytest.raises(ValueError, match="image count"):
        collect_cross_model(tmp_path, "test")


def test_missing_split_is_an_error_not_a_silent_skip(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", split="test")

    with pytest.raises(ValueError, match="no 'val' split"):
        collect_cross_model(tmp_path, "val")


def test_unsupported_model_name_is_rejected(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn")

    with pytest.raises(ValueError, match="unsupported model name"):
        collect_cross_model(tmp_path, "test", ["yolov9x"])


def test_detection_only_models_still_produce_a_table(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn", with_counting=False)

    payload = collect_cross_model(tmp_path, "test")

    assert "counting" not in payload["settings"]
    markdown = render_cross_model_markdown(payload, "title")
    assert "## Detection" in markdown
    assert "## Counting" not in markdown


def test_markdown_marks_deviating_rows_with_a_footnote(tmp_path: Path):
    _write_model(
        tmp_path,
        "faster_rcnn",
        deviations=("native detector score threshold lowered for full-range AP",),
    )
    _write_model(tmp_path, "yolov8n")

    markdown = render_cross_model_markdown(collect_cross_model(tmp_path, "test"), "title")

    assert "| faster_rcnn* |" in markdown
    assert "| yolov8n |" in markdown
    assert "* faster_rcnn: native detector score threshold lowered" in markdown


def test_write_cross_model_emits_both_artifacts(tmp_path: Path):
    _write_model(tmp_path, "faster_rcnn")
    _write_model(tmp_path, "yolov8n")

    json_path, markdown_path = write_cross_model(tmp_path, "test", "Reproduced comparison")

    assert json_path == tmp_path / "model_comparison_test.json"
    assert markdown_path == tmp_path / "model_comparison_test.md"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert len(payload["models"]) == 2
    assert "Reproduced comparison" in markdown_path.read_text(encoding="utf-8")


def test_empty_results_root_is_an_error(tmp_path: Path):
    with pytest.raises(ValueError, match="no evaluated model directories"):
        collect_cross_model(tmp_path, "test")


def test_variant_reads_the_variant_directory_where_one_exists(tmp_path: Path):
    _write_model(tmp_path, "fcos", map50=0.9)
    _write_model(tmp_path, "fcos-detector512", map50=0.8)
    _write_model(tmp_path, "yolov8n", map50=0.7)

    plain = collect_cross_model(tmp_path, "test")
    varied = collect_cross_model(tmp_path, "test", variant="detector512")

    # Without a variant the experiment directory is never folded in.
    assert {record["source"] for record in plain["models"]} == {"fcos", "yolov8n"}
    rows = {record["model"]: record for record in varied["models"]}
    assert rows["fcos"]["source"] == "fcos-detector512"
    assert rows["fcos"]["detection"]["map50"] == pytest.approx(0.8)
    assert rows["yolov8n"]["source"] == "yolov8n"
    assert varied["variant"] == "detector512"


def test_variant_without_any_variant_directory_is_an_error(tmp_path: Path):
    _write_model(tmp_path, "fcos")
    with pytest.raises(ValueError, match="no <model>-detector512 directory"):
        collect_cross_model(tmp_path, "test", variant="detector512")


def test_variant_tables_get_their_own_file_names(tmp_path: Path):
    _write_model(tmp_path, "fcos")
    _write_model(tmp_path, "fcos-detector512")

    json_path, markdown_path = write_cross_model(tmp_path, "test", "Variant", variant="detector512")

    assert json_path.name == "model_comparison_test_detector512.json"
    assert markdown_path.name == "model_comparison_test_detector512.md"
    assert b"\r" not in json_path.read_bytes()
