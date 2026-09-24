from pathlib import Path

import pytest

from cabbage_detection.data.official_layout import discover_official_dataset


def test_discovers_annotation_and_images_under_named_bundle(tmp_path: Path):
    bundle = tmp_path / "An annotated image dataset of cabbages for instance segmentation"
    (bundle / "images").mkdir(parents=True)
    (bundle / "annotation.json").write_text('{"images": [], "annotations": []}', encoding="utf-8")

    layout = discover_official_dataset(tmp_path)

    assert layout.download_root == tmp_path.resolve()
    assert layout.bundle_root == bundle.resolve()
    assert layout.images_root == (bundle / "images").resolve()
    assert layout.annotations_path == (bundle / "annotation.json").resolve()


def test_discovery_rejects_multiple_dataset_bundles(tmp_path: Path):
    for name in ("first", "second"):
        (tmp_path / name / "images").mkdir(parents=True)
        (tmp_path / name / "annotation.json").write_text(
            '{"images": [], "annotations": []}', encoding="utf-8"
        )
    with pytest.raises(ValueError, match="multiple.*annotation.json"):
        discover_official_dataset(tmp_path)


def test_discovery_rejects_annotation_without_sibling_images(tmp_path: Path):
    (tmp_path / "annotation.json").write_text(
        '{"images": [], "annotations": []}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="images"):
        discover_official_dataset(tmp_path)
