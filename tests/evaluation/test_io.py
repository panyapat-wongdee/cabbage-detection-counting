import json
from pathlib import Path

import pytest

from cabbage_detection.evaluation.io import (
    read_prediction_records,
    write_prediction_records,
    write_target_records,
)
from cabbage_detection.evaluation.records import Detection, PostprocessingMetadata, PredictionRecord


def test_read_prediction_records_accepts_schema_less_records(tmp_path: Path):
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "records": [{"image_id": "a", "detections": [{"box": [0, 0, 1, 1], "score": 0.9}]}],
            }
        ),
        encoding="utf-8",
    )

    records = read_prediction_records(path)

    assert records[0].image_id == "a"
    assert records[0].detections[0].score == 0.9


def test_read_prediction_records_rejects_legacy_version_marker(tmp_path: Path):
    path = tmp_path / "legacy-predictions.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [{"image_id": "a", "detections": []}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema-less"):
        read_prediction_records(path)


def test_read_prediction_records_rejects_legacy_list_format(tmp_path: Path):
    path = tmp_path / "legacy-predictions.json"
    path.write_text(json.dumps([{"box": [0, 0, 1, 1], "score": 0.9}]), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-less"):
        read_prediction_records(path)


def test_prediction_writer_round_trips_class_and_postprocessing_metadata(tmp_path: Path):
    path = tmp_path / "predictions.json"
    record = PredictionRecord(
        "a",
        (Detection((0, 0, 1, 1), 0.9, 1),),
        PostprocessingMetadata(
            "repository",
            "torchvision",
            0.5,
            0.5,
            100,
            0.0,
            (512, 512),
            "COCO_V1",
            "final",
            True,
        ),
    )
    write_prediction_records(path, [record])
    assert "schema_version" not in json.loads(path.read_text(encoding="utf-8"))
    assert json.loads(path.read_text(encoding="utf-8"))["records"][0]["postprocessing"]["native_nms_applied"] is True
    loaded = read_prediction_records(path)[0]
    assert loaded == record


def test_target_writer_is_schema_less(tmp_path: Path):
    path = tmp_path / "targets.json"
    write_target_records(path, [])
    assert json.loads(path.read_text(encoding="utf-8")) == {"records": []}
