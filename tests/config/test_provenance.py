from pathlib import Path

from cabbage_detection.config import load_config


def test_config_serializes_setting_sources(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model: {name: yolo11m, framework: ultralytics}
dataset: {format: yolo, root: dataset, split_manifest: splits.csv, yaml: dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
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
postprocess: {confidence_threshold: 0.5, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: results/run}
""",
        encoding="utf-8",
    )

    payload = load_config(path).to_dict()

    assert payload["provenance"]["training.epochs"] == "repository_protocol"
    assert payload["provenance"]["training.seed"] == "unknown"


def test_config_rejects_unknown_provenance_source(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model: {name: yolo11m, framework: ultralytics}
dataset: {format: yolo, root: dataset, split_manifest: splits.csv, yaml: dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
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
postprocess: {confidence_threshold: 0.5, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: results/run}
provenance: {training.epochs: historical_guess}
""",
        encoding="utf-8",
    )

    try:
        load_config(path)
    except ValueError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("invalid provenance source was accepted")


def test_framework_default_is_a_valid_provenance_source(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model: {name: yolo11m, framework: ultralytics}
dataset: {format: yolo, root: dataset, split_manifest: splits.csv, yaml: dataset/data.yaml, labels_dirname: null, class_names: [Cabbage]}
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
postprocess: {confidence_threshold: 0.5, iou_threshold: 0.5}
evaluation: {max_detections: 100, area_threshold: 0.0, confidence_owner: ultralytics, nms_owner: ultralytics, backend: repository_ap_101}
native_training: {patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}
artifacts: {output_dir: results/run}
provenance: {training.checkpoint_selection: framework_default}
""",
        encoding="utf-8",
    )

    assert load_config(path).provenance["training.checkpoint_selection"] == "framework_default"
