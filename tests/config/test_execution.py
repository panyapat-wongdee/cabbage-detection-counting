from pathlib import Path

import pytest

from cabbage_detection.config import load_config


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "execution.yaml"
    path.write_text(
        f"""
model: {{name: faster_rcnn, framework: torchvision}}
dataset: {{format: pascal_voc, root: dataset, split_manifest: splits.csv, yaml: null, labels_dirname: labels, class_names: [Cabbage]}}
initialization: {{weights: COCO_V1, num_classes: 2}}
training:
  image_size: [512, 512]
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  batch_size: 16
  epochs: 1
augmentation: {{profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}}
scheduler: {{name: step_lr, step_size: 1000, gamma: 0.1, final_lr_factor: null}}
postprocess: {{confidence_threshold: 0.5, iou_threshold: 0.5}}
evaluation: {{max_detections: 100, area_threshold: 0.0, confidence_owner: repository, nms_owner: repository, backend: pycocotools_coco_eval}}
native_training: {{patience: 500, cache: false, train_shuffle: true, validation_shuffle: false}}
artifacts: {{output_dir: results/run}}
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_execution_settings_are_explicit_and_serializable(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "execution: {device: cpu, workers: null, seed: null}\n"
            "checkpoint: {metric: null, mode: null}\n",
        )
    )
    assert config.execution.device == "cpu"
    assert config.execution.workers is None
    assert config.execution.seed is None
    assert config.checkpoint.metric is None
    assert config.to_dict()["execution"]["device"] == "cpu"


def test_checkpoint_policy_cannot_be_partially_defined(tmp_path: Path):
    with pytest.raises(ValueError, match="checkpoint metric and mode must be provided together"):
        load_config(_write_config(tmp_path, "checkpoint: {metric: map50, mode: null}\n"))


def test_checkpoint_resume_from_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match=r"unknown checkpoint key\(s\): resume_from"):
        load_config(_write_config(tmp_path, "checkpoint: {metric: null, mode: null, resume_from: null}\n"))


def test_checkpoint_policy_can_opt_in_to_per_epoch_artifacts(tmp_path: Path):
    config = load_config(
        _write_config(
            tmp_path,
            "checkpoint: {metric: train_loss, mode: min, retain_each_epoch: true}\n",
        )
    )
    assert config.checkpoint.retain_each_epoch is True
    assert config.to_dict()["checkpoint"]["retain_each_epoch"] is True


@pytest.mark.parametrize("device", ["tpu", "cuda:abc"])
def test_execution_rejects_unsupported_device(tmp_path: Path, device: str):
    with pytest.raises(ValueError, match="device"):
        load_config(_write_config(tmp_path, f"execution: {{device: {device}}}\n"))


def test_execution_rejects_negative_workers(tmp_path: Path):
    with pytest.raises(ValueError, match="workers"):
        load_config(_write_config(tmp_path, "execution: {device: cpu, workers: -1, seed: null}\n"))
