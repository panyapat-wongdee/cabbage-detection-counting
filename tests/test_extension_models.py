"""Models added after the study: scope, release boundary, configs, and NMS."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from cabbage_detection.config import load_config
from cabbage_detection.evaluation.cross_model import _study_scope
from cabbage_detection.models.ultralytics_adapter import _check_nms_owner
from cabbage_detection.release_artifacts import (
    EXTENSION_MODELS,
    RELEASE_MODELS,
    ReleaseValidationError,
    _validate_record,
    load_release_manifest,
    study_scope,
)

MANIFEST = Path("tests/release/fixture-manifest.json")


def test_every_model_has_exactly_one_scope():
    assert study_scope("fcos") == "paper_model"
    assert study_scope("ssdlite") == "supplementary_unreported"
    for model in EXTENSION_MODELS:
        assert study_scope(model) == "repository_extension"
        assert _study_scope(model) == "repository_extension"
    with pytest.raises(ValueError):
        study_scope("yolo99x")


def test_the_release_holds_every_model_and_only_named_variants():
    assert set(EXTENSION_MODELS) <= RELEASE_MODELS
    record = load_release_manifest(MANIFEST).checkpoints[0]

    def moved(model, run):
        return record.__class__(**{
            **record.__dict__,
            "model_name": model,
            "run_id": run,
            "framework": "ultralytics" if model.startswith("yolo") else record.framework,
            "study_scope": study_scope(model),
            "fine_tuned_license": "AGPL-3.0-only" if model.startswith("yolo") else record.fine_tuned_license,
            "source_path": f"runs/reproduced/{run}/checkpoints/best.pt",
            "metadata_path": f"runs/reproduced/{run}/metadata.json",
        })

    _validate_record(moved("yolo26n", "yolo26n"))
    _validate_record(moved("faster_rcnn", "faster_rcnn-detector512"))
    with pytest.raises(ReleaseValidationError, match="not a released run"):
        _validate_record(moved("faster_rcnn", "faster_rcnn-smoke"))
    mismatched = record.__class__(**{**record.__dict__, "source_path": "runs/reproduced/fcos/checkpoints/best.pt"})
    with pytest.raises(ReleaseValidationError, match="does not belong to run"):
        _validate_record(mismatched)


@pytest.mark.parametrize("model", EXTENSION_MODELS)
def test_extension_configs_follow_the_yolo11_protocol(model: str):
    config = load_config(Path(f"configs/ultralytics/{model}.yaml"))
    parent = load_config(Path(f"configs/ultralytics/yolo11{model[-1]}.yaml"))
    assert config.initialization.weights == f"{model}.pt"
    assert config.output_dir == Path(f"runs/reproduced/{model}")
    assert "historical_code" not in config.provenance.values()
    expected_nms = "none" if model.startswith("yolo26") else parent.evaluation.nms_owner
    assert config.evaluation.nms_owner == expected_nms
    for field in ("image_size", "optimizer", "learning_rate", "momentum", "weight_decay",
                  "batch_size", "epochs", "confidence_threshold", "iou_threshold",
                  "augmentation", "scheduler", "native_training", "execution", "checkpoint"):
        assert getattr(config, field) == getattr(parent, field), field


def _model_with_head(end2end):
    head = SimpleNamespace() if end2end is None else SimpleNamespace(end2end=end2end)
    return SimpleNamespace(model=SimpleNamespace(model=[head]))


def test_nms_owner_must_match_the_detection_head():
    _check_nms_owner("none", _model_with_head(True))           # YOLO26
    _check_nms_owner("ultralytics", _model_with_head(False))   # YOLOv8/11/12
    _check_nms_owner("none", _model_with_head(None))           # RT-DETR decoder
    with pytest.raises(ValueError, match="applies no NMS"):
        _check_nms_owner("ultralytics", _model_with_head(True))
    with pytest.raises(ValueError, match="needs NMS"):
        _check_nms_owner("none", _model_with_head(False))
