import json
from pathlib import Path

import pytest

from cabbage_detection.data.manifests import SplitManifest
from cabbage_detection.data.validation import validate_official_dataset


def _fixture(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("train.png", "val.png", "test.png"):
        (images / name).write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "train.png", "width": 10, "height": 10},
            {"id": 2, "file_name": "val.png", "width": 10, "height": 10},
            {"id": 3, "file_name": "test.png", "width": 10, "height": 10},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [2, 2, 2, 2]},
        ],
    }), encoding="utf-8")
    manifest = SplitManifest(
        (("train", "train"), ("val", "val"), ("test", "test")),
        "fixture", "",
    )
    return images, annotations, manifest


def test_validate_official_dataset_reports_split_and_annotation_counts(tmp_path: Path):
    report = validate_official_dataset(*_fixture(tmp_path))
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.image_count == 3
    assert report.annotation_count == 2
    assert len(report.annotations_sha256) == 64


def test_validate_official_dataset_rejects_overlap(tmp_path: Path):
    images, annotations, manifest = _fixture(tmp_path)
    duplicate = SplitManifest(manifest.rows + (("test", "train"),), "fixture", "")
    with pytest.raises(ValueError, match="multiple splits"):
        validate_official_dataset(images, annotations, duplicate)


def test_validate_official_dataset_finds_nested_official_image(tmp_path: Path):
    images = tmp_path / "images"
    nested = images / "OkinaSP" / "Kaizu" / "202010"
    nested.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png"):
        (nested / name).write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "path": "/OkinaSP/Kaizu/202010/a.png", "width": 10, "height": 10},
                    {"id": 2, "file_name": "b.png", "path": "/OkinaSP/Kaizu/202010/b.png", "width": 10, "height": 10},
                    {"id": 3, "file_name": "c.png", "path": "/OkinaSP/Kaizu/202010/c.png", "width": 10, "height": 10},
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = SplitManifest((("train", "a"), ("val", "b"), ("test", "c")), "fixture", "")

    report = validate_official_dataset(images, annotations, manifest)

    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
