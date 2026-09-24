from pathlib import Path

import pytest

from cabbage_detection.config import load_config


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_returns_serializable_experiment_config(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
model:
  name: yolo11m
  framework: ultralytics
dataset:
  format: yolo
  root: /dataset
  split_manifest: splits/fold1_recovered.csv
  yaml: /dataset/data.yaml
  labels_dirname: null
  class_names: [Cabbage]
initialization: {weights: yolo11m.pt, num_classes: 1}
training:
  image_size: [512, 512]
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  batch_size: 16
  epochs: 500
augmentation: {profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}
scheduler: {name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}
postprocess:
  confidence_threshold: 0.5
  iou_threshold: 0.5
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts:
  output_dir: runs/reproduced/run
""",
    )

    config = load_config(path)

    assert config.model_name == "yolo11m"
    assert config.framework == "ultralytics"
    assert config.image_size == (512, 512)
    assert config.to_dict()["training"]["epochs"] == 500


def test_load_config_rejects_unknown_top_level_key(tmp_path: Path):
    path = _write_config(tmp_path, "model: {}\nunexpected: true\n")

    with pytest.raises(ValueError, match="unknown"):
        load_config(path)


def test_load_config_rejects_publication_result_provenance(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
model: {name: yolo11m, framework: ultralytics}
dataset: {format: yolo, root: /dataset, split_manifest: splits.csv, yaml: /dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
initialization: {weights: yolo11m.pt, num_classes: 1}
training: {image_size: [512, 512], optimizer: SGD, learning_rate: 0.001, momentum: 0.9, weight_decay: 0.0005, batch_size: 16, epochs: 1}
augmentation: {profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}
scheduler: {name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}
postprocess: {confidence_threshold: 0.5, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 1, cache: false, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: results/run}
provenance: {training.epochs: pub-lished}
""".replace("pub-lished", "published"),
    )

    with pytest.raises(ValueError, match="provenance"):
        load_config(path)


def test_load_config_rejects_invalid_threshold(tmp_path: Path):
    path = _write_config(
        tmp_path,
        """
model: {name: yolo11m, framework: ultralytics}
dataset: {format: yolo, root: /dataset, split_manifest: splits/fold1_recovered.csv, yaml: /dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
initialization: {weights: yolo11m.pt, num_classes: 1}
training:
  image_size: [512, 512]
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  batch_size: 16
  epochs: 500
augmentation: {profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}
scheduler: {name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}
postprocess: {confidence_threshold: 1.2, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: runs/reproduced/run}
""",
    )

    with pytest.raises(ValueError, match="confidence_threshold"):
        load_config(path)


def test_counting_iou_threshold_defaults_to_the_nms_threshold(tmp_path: Path):
    config = load_config(Path("configs/ultralytics/yolov8n.yaml"))
    assert config.evaluation.counting_iou_threshold is None
    assert config.counting_iou_threshold == config.iou_threshold


def test_counting_iou_threshold_can_be_set_independently(tmp_path: Path):
    source = Path("configs/ultralytics/yolov8n.yaml").read_text(encoding="utf-8")
    source = source.replace("  backend: repository_ap_101", "  backend: repository_ap_101\n  counting_iou_threshold: 0.25")
    path = _write_config(tmp_path, source)
    config = load_config(path)
    assert config.iou_threshold == 0.5
    assert config.counting_iou_threshold == 0.25
    assert config.to_dict()["evaluation"]["counting_iou_threshold"] == 0.25


def test_counting_iou_threshold_must_be_a_fraction(tmp_path: Path):
    source = Path("configs/ultralytics/yolov8n.yaml").read_text(encoding="utf-8")
    source = source.replace("  backend: repository_ap_101", "  backend: repository_ap_101\n  counting_iou_threshold: 1.5")
    with pytest.raises(ValueError, match="counting_iou_threshold"):
        load_config(_write_config(tmp_path, source))
