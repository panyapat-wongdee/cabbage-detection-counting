import json
from pathlib import Path

from cabbage_detection.data.audit import audit_dataset_migration
from cabbage_detection.data.coco import read_coco_detection
from cabbage_detection.data.manifests import SplitManifest, _digest, write_manifest
from cabbage_detection.data.yolo import prepare_yolo_dataset


def test_audit_reports_exact_membership_and_counts(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (images / name).write_bytes(b"image")
    annotations = tmp_path / "labels.json"
    annotations.write_text(json.dumps({
        "images": [
            {"id": 1, "file_name": "a.png", "width": 10, "height": 10},
            {"id": 2, "file_name": "b.png", "width": 10, "height": 10},
            {"id": 3, "file_name": "c.png", "width": 10, "height": 10},
        ],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
    }), encoding="utf-8")
    rows = (("train", "a"), ("val", "b"), ("test", "c"))
    manifest = tmp_path / "fold.csv"
    write_manifest(SplitManifest(rows, "fixture", _digest(rows)), manifest)
    generated = tmp_path / "generated"
    prepare_yolo_dataset(read_coco_detection(annotations), SplitManifest(rows, "fixture", _digest(rows)), generated, images)

    report = audit_dataset_migration(images, annotations, manifest, generated)

    assert report.split_counts == {"train": 1, "val": 1, "test": 1}
    assert report.missing_ids == ()
    assert report.overlapping_ids == ()
    assert report.annotation_count_mismatches == ()
    assert report.maximum_yolo_round_trip_error <= 1e-6


def test_audit_resolves_nested_official_image_path(tmp_path: Path):
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
                "annotations": [{"id": 1, "image_id": 1, "bbox": [1, 1, 2, 2]}],
            }
        ),
        encoding="utf-8",
    )
    rows = (("train", "a"), ("val", "b"), ("test", "c"))
    manifest = tmp_path / "fold.csv"
    write_manifest(SplitManifest(rows, "fixture", _digest(rows)), manifest)
    generated = tmp_path / "generated"
    source = read_coco_detection(annotations)
    prepare_yolo_dataset(source, SplitManifest(rows, "fixture", "digest"), generated, images)

    report = audit_dataset_migration(images, annotations, manifest, generated)

    assert report.missing_ids == ()
    assert report.annotation_count_mismatches == ()
