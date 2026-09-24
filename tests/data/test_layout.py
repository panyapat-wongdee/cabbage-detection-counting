from pathlib import Path

import pytest

from cabbage_detection.data.layout import validate_dataset_root


def test_validate_dataset_root_finds_image_label_pairs(tmp_path: Path):
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        (tmp_path / split / "images" / "sample.png").write_bytes(b"image")
        (tmp_path / split / "labels" / "sample.txt").write_text("0 0.5 0.5 1 1\n")

    layout = validate_dataset_root(tmp_path)

    assert layout.root == tmp_path
    assert layout.image_ids("train") == ("sample",)


def test_validate_dataset_root_rejects_missing_label(tmp_path: Path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir()
    (tmp_path / "train" / "images" / "sample.png").write_bytes(b"image")

    with pytest.raises(ValueError, match="label"):
        validate_dataset_root(tmp_path)
