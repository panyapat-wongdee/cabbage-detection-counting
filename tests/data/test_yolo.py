import json
import os
from pathlib import Path

import pytest
import yaml

from cabbage_detection.data.coco import read_coco_detection, xyxy_to_yolo
from cabbage_detection.data.manifests import SplitManifest
from cabbage_detection.data.yolo import prepare_yolo_dataset


def test_xyxy_to_yolo_normalizes_coordinates():
    assert xyxy_to_yolo((0, 0, 100, 50), 200, 100) == pytest.approx((0.25, 0.25, 0.5, 0.5))


def test_xyxy_to_yolo_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        xyxy_to_yolo((0, 0, 1, 1), 0, 100)


def test_prepare_yolo_uses_hardlinks_and_reports_round_trip(tmp_path: Path):
    source_root = tmp_path / "images"
    source_root.mkdir()
    (source_root / "a.png").write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
    }), encoding="utf-8")
    source = read_coco_detection(annotations)
    manifest = SplitManifest((("train", "a2"),), "fixture", "digest")
    # This fixture intentionally tests the source validation error path below.
    with pytest.raises(ValueError, match="manifest image_id"):
        prepare_yolo_dataset(source, manifest, tmp_path / "prepared", source_root)


def test_prepare_yolo_uses_hardlinks_for_valid_manifest(tmp_path: Path):
    source_root = tmp_path / "images"
    source_root.mkdir()
    (source_root / "a.png").write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
    }), encoding="utf-8")
    source = read_coco_detection(annotations)
    manifest = SplitManifest((("train", "a"),), "fixture", "digest")
    report = prepare_yolo_dataset(source, manifest, tmp_path / "prepared", source_root)
    assert os.path.samefile(source_root / "a.png", tmp_path / "prepared/train/images/a.png")
    assert report.maximum_round_trip_error <= 1e-6


def test_prepare_yolo_data_yaml_resolves_paths_from_generated_root(tmp_path: Path):
    source_root = tmp_path / "images"
    source_root.mkdir()
    (source_root / "a.png").write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    source = read_coco_detection(annotations)
    manifest = SplitManifest((("train", "a"),), "fixture", "digest")
    output = tmp_path / "prepared"

    prepare_yolo_dataset(source, manifest, output, source_root)

    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    dataset_root = Path(data["path"])
    assert dataset_root == output.resolve()
    assert (dataset_root / data["train"]).is_dir()


def test_prepare_yolo_resolves_nested_official_image_path(tmp_path: Path):
    source_root = tmp_path / "images"
    nested = source_root / "OkinaSP" / "Kaizu" / "202010"
    nested.mkdir(parents=True)
    image_path = nested / "a.png"
    image_path.write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "a.png",
                        "path": "/OkinaSP/Kaizu/202010/a.png",
                        "width": 10,
                        "height": 10,
                    }
                ],
                "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    source = read_coco_detection(annotations)
    manifest = SplitManifest((("train", "a"),), "fixture", "digest")

    prepare_yolo_dataset(source, manifest, tmp_path / "prepared", source_root)

    assert os.path.samefile(image_path, tmp_path / "prepared/train/images/a.png")
