"""Ground-truth target construction for canonical evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cabbage_detection.config import load_config
from cabbage_detection.data.manifests import SplitManifest, _digest, write_manifest
from cabbage_detection.evaluation import targets as targets_module
from cabbage_detection.evaluation.targets import (
    build_target_records,
    load_evaluation_dataset,
    split_image_ids,
)

CONFIG_TEMPLATE = """
model: {{name: {model}, framework: {framework}}}
dataset: {{format: {format}, root: dataset, split_manifest: splits.csv, yaml: {yaml}, labels_dirname: null, class_names: [Cabbage]}}
initialization: {{weights: {weights}, num_classes: {classes}}}
training:
  image_size: [512, 512]
  optimizer: SGD
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0005
  batch_size: 1
  epochs: 1
augmentation: {{profile: reproduction_augmented, horizontal_flip: 0.5, vertical_flip: 0.5, translate: 0.1, scale: 0.1, degrees: 15.0, hsv_h: 0.1, hsv_s: 0.05, hsv_v: 0.05, mosaic: 0.0, normalize: true, border_value: 114}}
scheduler: {{name: constant, step_size: null, gamma: null, final_lr_factor: 1.0}}
postprocess: {{confidence_threshold: 0.5, iou_threshold: 0.5}}
evaluation: {{max_detections: 100, area_threshold: 0.0, confidence_owner: {owner}, nms_owner: {owner}, backend: repository_ap_101}}
native_training: {{patience: 500, cache: true, train_shuffle: true, validation_shuffle: false}}
artifacts: {{output_dir: run}}
"""


def _config(tmp_path: Path, framework: str):
    if framework == "ultralytics":
        text = CONFIG_TEMPLATE.format(
            model="yolo11n", framework="ultralytics", format="yolo", yaml="dataset/data.yaml",
            weights="yolo11n.pt", owner="ultralytics", classes=1,
        )
    else:
        text = CONFIG_TEMPLATE.format(
            model="faster_rcnn", framework="torchvision", format="coco", yaml="null",
            weights="COCO_V1", owner="torchvision", classes=2,
        )
    path = tmp_path / f"{framework}.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _annotations(tmp_path: Path) -> Path:
    path = tmp_path / "annotation.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 7, "file_name": "images/img000.png", "width": 256, "height": 128},
                    {"id": 8, "file_name": "images/img001.png", "width": 256, "height": 128},
                ],
                "annotations": [
                    {"id": 1, "image_id": 7, "category_id": 1, "bbox": [10, 20, 30, 40]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ultralytics_targets_stay_in_original_pixels(tmp_path: Path):
    records = build_target_records(_config(tmp_path, "ultralytics"), _annotations(tmp_path), ["img000"])
    assert len(records) == 1
    assert records[0].image_id == "img000"
    assert records[0].boxes[0].coordinates == (10.0, 20.0, 40.0, 60.0)


def test_torchvision_targets_are_rescaled_to_the_configured_image_size(tmp_path: Path):
    records = build_target_records(_config(tmp_path, "torchvision"), _annotations(tmp_path), ["img000"])
    # 512/256 horizontally and 512/128 vertically, matching the eval-time resize.
    assert records[0].boxes[0].coordinates == (20.0, 80.0, 80.0, 240.0)


def test_images_without_annotations_produce_empty_targets(tmp_path: Path):
    records = build_target_records(_config(tmp_path, "ultralytics"), _annotations(tmp_path), ["img001"])
    assert records[0].boxes == ()


def test_target_order_follows_the_requested_image_ids(tmp_path: Path):
    records = build_target_records(
        _config(tmp_path, "ultralytics"), _annotations(tmp_path), ["img001", "img000"]
    )
    assert [record.image_id for record in records] == ["img001", "img000"]


def test_unknown_image_id_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="missing from COCO image table"):
        build_target_records(_config(tmp_path, "ultralytics"), _annotations(tmp_path), ["absent"])


def _manifest(tmp_path: Path) -> Path:
    rows = (("train", "a"), ("test", "b"), ("test", "c"))
    path = tmp_path / "splits.csv"
    write_manifest(SplitManifest(rows=rows, provenance="test", digest=_digest(rows)), path)
    return path


def test_split_image_ids_preserve_manifest_order(tmp_path: Path):
    assert split_image_ids(_manifest(tmp_path), "test") == ("b", "c")


def test_split_image_ids_reject_an_absent_split(tmp_path: Path):
    with pytest.raises(ValueError, match="not present in the manifest"):
        split_image_ids(_manifest(tmp_path), "val")


def test_load_evaluation_dataset_parses_the_annotations_once(tmp_path: Path, monkeypatch):
    """The split loop reuses one parse instead of re-reading per split."""
    bundle = tmp_path / "download" / "cabbage"
    (bundle / "images").mkdir(parents=True)
    (bundle / "images" / "img000.png").write_bytes(b"image")
    (bundle / "annotation.json").write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "images/img000.png", "width": 100, "height": 50}],
                "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    reads: list[Path] = []
    original = targets_module.read_coco_detection

    def counting_read(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(targets_module, "read_coco_detection", counting_read)
    dataset = load_evaluation_dataset(tmp_path / "download")

    assert len(reads) == 1
    assert dataset.image_path("img000").name == "img000.png"
    # Building targets for several splits must not read the file again.
    build_target_records(_config(tmp_path, "ultralytics"), dataset.coco, ["img000"])
    build_target_records(_config(tmp_path, "ultralytics"), dataset.coco, ["img000"])
    assert len(reads) == 1


def test_evaluation_dataset_rejects_an_unknown_image_id(tmp_path: Path):
    bundle = tmp_path / "download" / "cabbage"
    (bundle / "images").mkdir(parents=True)
    (bundle / "images" / "img000.png").write_bytes(b"image")
    (bundle / "annotation.json").write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "images/img000.png", "width": 100, "height": 50}],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_evaluation_dataset(tmp_path / "download")
    with pytest.raises(ValueError, match="missing from COCO image table"):
        dataset.image_path("absent")
