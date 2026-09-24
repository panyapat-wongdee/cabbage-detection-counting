from pathlib import Path

import pytest

from cabbage_detection.data.torchvision_dataset import PascalDetectionDataset, read_pascal_labels


def test_pascal_label_parser_reads_xyxy_and_maps_cabbage_to_torchvision_label(tmp_path: Path):
    label_path = tmp_path / "sample.txt"
    label_path.write_text("0 1 2 11 22\n", encoding="utf-8")
    assert read_pascal_labels(label_path) == ((1.0, 2.0, 11.0, 22.0),)


@pytest.mark.parametrize(
    "content, message",
    [
        ("1 1 2 11 22\n", "class id"),
        ("0 1 2 3\n", "five values"),
        ("0 4 2 1 22\n", "ordered"),
        ("0 -1 2 11 22\n", "non-negative"),
    ],
)
def test_pascal_label_parser_rejects_invalid_rows(tmp_path: Path, content: str, message: str):
    path = tmp_path / "bad.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_pascal_labels(path)


def test_pascal_dataset_requires_image_and_label_pairs(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "sample.png").write_bytes(b"not decoded")
    with pytest.raises(FileNotFoundError, match="sample.txt"):
        PascalDetectionDataset(tmp_path / "images", tmp_path / "labels", ("sample",))


def test_pascal_dataset_ram_cache_reuses_decoded_image(tmp_path: Path):
    pytest.importorskip("torch")
    from PIL import Image

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(images / "sample.png")
    (labels / "sample.txt").write_text("0 1 2 11 8\n", encoding="utf-8")

    dataset = PascalDetectionDataset(images, labels, ("sample",), cache=True)

    assert dataset.cache_enabled is True
    assert dataset.cached_image_count == 0
    dataset[0]
    assert dataset.cached_image_count == 1
    dataset[0]
    assert dataset.cached_image_count == 1
