from pathlib import Path

from cabbage_detection.data.layout import validate_dataset_root
from cabbage_detection.data.manifests import build_manifest, read_manifest, write_manifest


def _layout(tmp_path: Path):
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        (tmp_path / split / "images" / f"{split}_1.png").write_bytes(b"image")
        (tmp_path / split / "labels" / f"{split}_1.txt").write_text("", encoding="utf-8")
    return validate_dataset_root(tmp_path)


def test_manifest_round_trip_has_stable_digest(tmp_path: Path):
    manifest = build_manifest(_layout(tmp_path), provenance="recovered-local-layout")
    output = tmp_path / "manifest.csv"

    write_manifest(manifest, output)
    loaded = read_manifest(output)

    assert loaded.provenance == "recovered-local-layout"
    assert loaded.digest == manifest.digest
    assert len(loaded.rows) == 3
