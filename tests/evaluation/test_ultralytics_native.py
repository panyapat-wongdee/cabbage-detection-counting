from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cabbage_detection.config import (
    AugmentationConfig,
    CheckpointConfig,
    DatasetConfig,
    EvaluationConfig,
    ExecutionConfig,
    ExperimentConfig,
    InitializationConfig,
    NativeTrainingConfig,
    SchedulerConfig,
)
from cabbage_detection.evaluation.ultralytics_native import evaluate_ultralytics_native


def _native_config(tmp_path: Path):
    return ExperimentConfig(
        model_name="yolo11n",
        framework="ultralytics",
        dataset=DatasetConfig("yolo", Path("dataset"), Path("manifest.csv"), Path("data.yaml"), None, ("Cabbage",)),
        initialization=InitializationConfig("yolo11n.pt", 1),
        image_size=(512, 512), optimizer="SGD", learning_rate=0.001, momentum=0.9,
        weight_decay=0.0005, batch_size=16, epochs=3, confidence_threshold=0.5,
        iou_threshold=0.5,
        augmentation=AugmentationConfig("recovered_no_augmentation", 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 114),
        scheduler=SchedulerConfig("constant", None, None, 1.0),
        evaluation=EvaluationConfig(100, 0.0, "ultralytics", "ultralytics", "repository_ap_101"),
        native_training=NativeTrainingConfig(500, True, True, False),
        output_dir=tmp_path / "run", execution=ExecutionConfig("cpu", 8, 17),
        checkpoint=CheckpointConfig(None, None),
    )


def test_native_validation_passes_explicit_paper_arguments(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    dataset_yaml = tmp_path / "data.yaml"
    checkpoint.write_bytes(b"checkpoint")
    dataset_yaml.write_text("path: dataset\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    class FakeModel:
        def val(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(box=SimpleNamespace(map50=0.61, map=0.37))

    monkeypatch.setattr("cabbage_detection.evaluation.ultralytics_native._load_model", lambda config, path: FakeModel())
    output = evaluate_ultralytics_native(
        _native_config(tmp_path), checkpoint, dataset_yaml, "test", tmp_path / "evaluation"
    )

    assert output.name == "native_metrics.json"
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert payload["metrics"] == {"map50": 0.61, "map50_95": 0.37}
    assert payload["evaluation"]["backend"] == "ultralytics_native_val"
    assert calls == [
        {
            "data": str(dataset_yaml),
            "split": "test",
            "conf": 0.5,
            "iou": 0.5,
            "max_det": 100,
            "batch": 16,
            "imgsz": 512,
            "device": "cpu",
            "project": str((tmp_path / "evaluation").resolve()),
            "name": "framework",
            "plots": False,
            "save_json": False,
        }
    ]
    assert (tmp_path / "evaluation" / "evaluation_metadata.json").exists()


@pytest.mark.parametrize("split", ["invalid", "val/../test"])
def test_native_validation_rejects_invalid_split(tmp_path: Path, split: str):
    with pytest.raises(ValueError, match="split"):
        evaluate_ultralytics_native(_native_config(tmp_path), tmp_path / "best.pt", tmp_path / "data.yaml", split, tmp_path / "out")


def test_native_validation_rejects_wrong_framework(tmp_path: Path):
    config = replace(_native_config(tmp_path), framework="torchvision")
    with pytest.raises(ValueError, match="Ultralytics"):
        evaluate_ultralytics_native(config, tmp_path / "best.pt", tmp_path / "data.yaml", "test", tmp_path / "out")


def test_native_validation_rejects_missing_inputs(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        evaluate_ultralytics_native(_native_config(tmp_path), tmp_path / "best.pt", tmp_path / "data.yaml", "test", tmp_path / "out")
