import json
import warnings
from pathlib import Path

import pytest

from cabbage_detection.data.torchvision_dataset import CabbageDetectionDataset, collate_detection_batch


def _fixture(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    images = tmp_path / "images"
    images.mkdir()
    (images / "second.png").write_bytes(b"not-decoded-in-this-test")
    (images / "first.png").write_bytes(b"not-decoded-in-this-test")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 2, "file_name": "second.png", "width": 20, "height": 10},
                    {"id": 1, "file_name": "first.png", "width": 20, "height": 10},
                ],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )
    return labels, ("first", "second")


def test_dataset_preserves_manifest_order_and_index_metadata(tmp_path: Path):
    labels, image_ids = _fixture(tmp_path)
    dataset = CabbageDetectionDataset(tmp_path / "images", labels, image_ids)
    assert dataset.image_ids == image_ids
    assert dataset.samples[0].image_id == "first"
    assert dataset.samples[0].annotations[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_dataset_rejects_manifest_identifier_without_coco_image(tmp_path: Path):
    labels, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest image_id.*missing"):
        CabbageDetectionDataset(tmp_path / "images", labels, ("missing",))


def test_collate_returns_images_and_targets_separately():
    images = ["image-a", "image-b"]
    targets = [{"image_id": 1}, {"image_id": 2}]
    assert collate_detection_batch(list(zip(images, targets))) == (images, targets)


def test_dataset_loads_rgb_tensor_and_detection_target(tmp_path: Path):
    pytest.importorskip("torch")
    from PIL import Image

    labels, image_ids = _fixture(tmp_path)
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(tmp_path / "images" / "first.png")
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(tmp_path / "images" / "second.png")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image, target = CabbageDetectionDataset(tmp_path / "images", labels, image_ids)[0]
    assert not any("non-writable" in str(item.message) for item in caught)
    assert tuple(image.shape) == (3, 10, 20)
    assert target["boxes"].tolist() == [[1.0, 2.0, 4.0, 6.0]]
    assert target["labels"].tolist() == [1]


def test_dataset_ram_cache_reuses_decoded_image(tmp_path: Path):
    pytest.importorskip("torch")
    from PIL import Image

    labels, image_ids = _fixture(tmp_path)
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(tmp_path / "images" / "first.png")
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(tmp_path / "images" / "second.png")

    dataset = CabbageDetectionDataset(tmp_path / "images", labels, image_ids, cache=True)

    assert dataset.cache_enabled is True
    assert dataset.cached_image_count == 0
    dataset[0]
    assert dataset.cached_image_count == 1
    dataset[0]
    assert dataset.cached_image_count == 1


def test_dataset_resolves_nested_official_image_path(tmp_path: Path):
    images = tmp_path / "images"
    nested = images / "OkinaSP" / "Kaizu" / "202010"
    nested.mkdir(parents=True)
    image_path = nested / "a.png"
    image_path.write_bytes(b"image")
    labels = tmp_path / "labels.json"
    labels.write_text(
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
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )

    dataset = CabbageDetectionDataset(images, labels, ("a",))

    assert dataset.samples[0].path == image_path
