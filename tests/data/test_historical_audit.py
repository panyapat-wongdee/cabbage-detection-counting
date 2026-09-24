from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from cabbage_detection.data.historical_audit import audit_historical_inputs


def _write_manifest(path: Path) -> None:
    path.write_text(
        "split,image_id,provenance,digest\ntrain,img1,test,PLACEHOLDER\nval,img2,test,PLACEHOLDER\n",
        encoding="utf-8",
    )
    import hashlib

    digest = hashlib.sha256("train,img1\nval,img2".encode()).hexdigest()
    path.write_text(path.read_text().replace("PLACEHOLDER", digest), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    coco_root = tmp_path / "coco"
    bundle = coco_root / "bundle"
    (bundle / "images").mkdir(parents=True)
    pascal_root = tmp_path / "pascal"
    for split in ("train", "val", "test"):
        (pascal_root / split / "images").mkdir(parents=True)
        (pascal_root / split / "labels").mkdir(parents=True)
    image1 = bundle / "images" / "img1.png"
    image2 = bundle / "images" / "img2.png"
    Image.new("RGB", (20, 20), "white").save(image1)
    Image.new("RGB", (20, 20), "black").save(image2)
    Image.open(image1).save(pascal_root / "train" / "images" / "img1.png")
    Image.open(image2).save(pascal_root / "val" / "images" / "img2.png")
    (pascal_root / "train" / "labels" / "img1.txt").write_text("0 1 2 10 12\n", encoding="utf-8")
    (pascal_root / "val" / "labels" / "img2.txt").write_text("", encoding="utf-8")
    annotation = {
        "images": [
            {"id": 1, "file_name": "img1.png", "path": "img1.png", "width": 20, "height": 20},
            {"id": 2, "file_name": "img2.png", "path": "img2.png", "width": 20, "height": 20},
        ],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 9, 10]}],
        "categories": [{"id": 1, "name": "Cabbage"}],
    }
    (bundle / "annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    return coco_root, pascal_root, manifest


def test_historical_inputs_audit_reports_equivalence_and_empty_image(tmp_path: Path) -> None:
    coco_root, pascal_root, manifest = _fixture(tmp_path)
    report = audit_historical_inputs(coco_root, pascal_root, manifest)
    assert report["equivalent"] is True
    assert report["checked_images"] == 2
    assert report["empty_images"] == {"coco": 1, "pascal": 1}


def test_historical_inputs_audit_reports_box_mismatch(tmp_path: Path) -> None:
    coco_root, pascal_root, manifest = _fixture(tmp_path)
    (pascal_root / "train" / "labels" / "img1.txt").write_text("0 1 2 11 12\n", encoding="utf-8")
    report = audit_historical_inputs(coco_root, pascal_root, manifest)
    assert report["equivalent"] is False
    assert report["mismatch_count"] == 1
    assert "box_coordinates_mismatch" in report["mismatches"][0]["reasons"]


def test_historical_inputs_audit_reports_category_mismatch(tmp_path: Path) -> None:
    coco_root, pascal_root, manifest = _fixture(tmp_path)
    annotation_path = coco_root / "bundle" / "annotation.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["category_id"] = 2
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report = audit_historical_inputs(coco_root, pascal_root, manifest)

    assert report["equivalent"] is False
    assert report["mismatch_count"] == 1
    assert "category_id_mismatch" in report["mismatches"][0]["reasons"]
