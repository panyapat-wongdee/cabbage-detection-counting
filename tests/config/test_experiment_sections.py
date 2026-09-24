from pathlib import Path

import pytest

from cabbage_detection.config import load_config


def _config_yaml(**overrides: str) -> str:
    values = {
        "dataset_format": "yolo",
        "dataset_root": "data/fold1_yolo",
        "dataset_yaml": "data/fold1_yolo/data.yaml",
        "labels_dirname": "null",
        "weights": "yolov8n.pt",
        "num_classes": "1",
        "profile": "recovered_no_augmentation",
        "confidence_owner": "ultralytics",
        "nms_owner": "ultralytics",
        "backend": "repository_ap_101",
        "result_status": "reproduction_candidate",
    }
    values.update(overrides)
    return f"""
model: {{name: yolov8n, framework: ultralytics}}
dataset:
  format: {values['dataset_format']}
  root: {values['dataset_root']}
  split_manifest: splits/fold1_recovered.csv
  yaml: {values['dataset_yaml']}
  labels_dirname: {values['labels_dirname']}
  class_names: [Cabbage]
initialization: {{weights: {values['weights']}, num_classes: {values['num_classes']}}}
training: {{image_size: [512, 512], optimizer: SGD, learning_rate: 0.001, momentum: 0.9, weight_decay: 0.0005, batch_size: 16, epochs: 500}}
augmentation: {{profile: {values['profile']}, horizontal_flip: 0.0, vertical_flip: 0.0, translate: 0.0, scale: 0.0, degrees: 0.0, hsv_h: 0.0, hsv_s: 0.0, hsv_v: 0.0, mosaic: 0.0, normalize: false, border_value: 114}}
scheduler: {{name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}}
postprocess: {{confidence_threshold: 0.5, iou_threshold: 0.5}}
evaluation: {{max_detections: 100, area_threshold: 0.0, confidence_owner: {values['confidence_owner']}, nms_owner: {values['nms_owner']}, backend: {values['backend']}}}
native_training: {{patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}}
artifacts: {{output_dir: runs/reproduced/yolov8n-noaug, result_status: {values['result_status']}}}
execution: {{device: cuda:1, workers: 8, seed: null}}
checkpoint: {{metric: null, mode: null}}
provenance: {{augmentation.profile: historical_code, execution.workers: historical_code, execution.seed: unknown}}
""".strip()


def test_load_config_exposes_explicit_experiment_sections(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(_config_yaml(), encoding="utf-8")

    config = load_config(path)

    assert config.dataset.format == "yolo"
    assert config.dataset.yaml == Path("data/fold1_yolo/data.yaml")
    assert config.initialization.weights == "yolov8n.pt"
    assert config.initialization.num_classes == 1
    assert config.augmentation.profile == "recovered_no_augmentation"
    assert config.scheduler.final_lr_factor == 1.0
    assert config.evaluation.max_detections == 100
    assert config.native_training.patience == 500


@pytest.mark.parametrize(
        ("overrides", "message"),
    [
        ({"dataset_yaml": "null"}, "dataset.yaml"),
        ({"confidence_owner": "invalid"}, "confidence owner"),
        ({"nms_owner": "invalid"}, "NMS owner"),
    ],
)
def test_load_config_rejects_invalid_framework_contracts(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(_config_yaml(**overrides), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_rejects_zero_max_detections(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(_config_yaml().replace("max_detections: 100", "max_detections: 0"), encoding="utf-8")

    with pytest.raises(ValueError, match="max_detections"):
        load_config(path)


def test_load_config_rejects_missing_evaluation_backend(tmp_path: Path) -> None:
    path = tmp_path / "missing-backend.yaml"
    path.write_text(_config_yaml().replace(", backend: repository_ap_101", ""), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluation.backend"):
        load_config(path)


def test_artifacts_result_status_accepts_only_trainable_states(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(_config_yaml(result_status="smoke"), encoding="utf-8")
    assert load_config(smoke).result_status == "smoke"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(_config_yaml(result_status="reproduced_verified"), encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts.result_status"):
        load_config(invalid)
