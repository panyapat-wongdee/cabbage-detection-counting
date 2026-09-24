import json
from pathlib import Path

import pytest

from cabbage_detection.data.coco import (
    CocoImage,
    coco_xywh_to_xyxy,
    read_coco_detection,
    resolve_coco_image_path,
)


def test_read_coco_detection_and_convert_box(tmp_path: Path):
    annotations = tmp_path / "labels.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.png", "width": 200, "height": 100}],
                "annotations": [{"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}],
                "categories": [{"id": 1, "name": "cabbage"}],
            }
        ),
        encoding="utf-8",
    )

    dataset = read_coco_detection(annotations)

    assert dataset.images[1].file_name == "a.png"
    assert dataset.annotations[0].bbox == (10.0, 20.0, 30.0, 40.0)
    assert coco_xywh_to_xyxy(dataset.annotations[0].bbox) == (10.0, 20.0, 40.0, 60.0)


def test_read_coco_rejects_unknown_image_reference(tmp_path: Path):
    annotations = tmp_path / "labels.json"
    annotations.write_text(
        json.dumps({"images": [], "annotations": [{"image_id": 9, "bbox": [0, 0, 1, 1]}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="image_id"):
        read_coco_detection(annotations)


def test_reads_official_path_and_resolves_nested_image(tmp_path: Path):
    images = tmp_path / "images"
    nested = images / "OkinaSP" / "Kaizu" / "202010" / "img032.png"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"png")
    annotation = tmp_path / "labels.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "img032.png",
                        "path": "/OkinaSP/Kaizu/202010/img032.png",
                        "width": 515,
                        "height": 515,
                    }
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    image = read_coco_detection(annotation).images[1]

    assert image.relative_path == Path("OkinaSP/Kaizu/202010/img032.png")
    assert resolve_coco_image_path(images, image) == nested.resolve()


@pytest.mark.parametrize("path", ["/../../secret.png", "../secret.png", "C:/secret.png"])
def test_rejects_unsafe_coco_image_path(tmp_path: Path, path: str):
    annotation = tmp_path / "labels.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "path": path, "width": 10, "height": 10}
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative path"):
        read_coco_detection(annotation)


def test_file_name_fallback_remains_supported(tmp_path: Path):
    image = CocoImage(1, "a.png", 10, 10)

    assert resolve_coco_image_path(tmp_path, image) == (tmp_path / "a.png").resolve()
