"""Evidence assembly that backs a reproduction promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from cabbage_detection.release_artifacts import DATASET_DOI
from cabbage_detection.reproduction_evidence import (
    DATASET_VERSION,
    build_dataset_evidence,
    build_evaluation_evidence,
    build_pretrained_evidence,
    read_weights_registry,
    resolve_pretrained_artifact,
    stage_run_evidence,
)

BASE_BYTES = b"pretend base weights"
BASE_DIGEST = hashlib.sha256(BASE_BYTES).hexdigest()


def _registry(filename: str) -> dict[str, object]:
    return {
        "yolov8n": {
            "framework": "ultralytics",
            "framework_version": "8.4.9",
            "pretrained": {
                "identifier": "YOLOv8n",
                "expected_filename": filename,
                "source_url": "https://example.invalid/yolov8n.pt",
                "software_license": "AGPL-3.0-only",
                "training_dataset": "MS COCO",
                "redistributed": False,
            },
        }
    }


def _official_dataset(root: Path, annotation: bytes = b'{"images": []}') -> Path:
    bundle = root / "An annotated image dataset of cabbages"
    (bundle / "images").mkdir(parents=True)
    (bundle / "annotation.json").write_bytes(annotation)
    return root


def _results(root: Path, split: str = "test") -> Path:
    split_dir = root / split
    (split_dir / "records").mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps({"detection_backends": ["cabbage_detection.ap_101"], "splits": {}}),
        encoding="utf-8",
    )
    (split_dir / "detection.json").write_text(
        json.dumps(
            {
                "scope": {"image_count": 92},
                "metrics": {"map50": 0.9, "map50_95": 0.7, "backend": "cabbage_detection.ap_101"},
                "evaluation": {"ap_iou_threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "counting.json").write_text(
        json.dumps(
            {
                "scope": {"image_count": 92},
                "metrics": {"f1": 0.95, "true_positives": 10},
                "evaluation": {"iou_threshold": 0.5, "confidence_threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "metadata.json").write_text(
        json.dumps(
            {
                "deviations": ["native detector score threshold lowered for full-range AP"],
                "git_revision": "b" * 40,
                "command": ["scripts/evaluate.py"],
                "inputs": {"predictions": {"sha256": "c" * 64}},
            }
        ),
        encoding="utf-8",
    )
    (split_dir / "records" / "predictions.json").write_text("[]", encoding="utf-8")
    return root


def test_dataset_evidence_hashes_the_official_annotation_and_manifest(tmp_path: Path):
    dataset_root = _official_dataset(tmp_path / "download", b'{"images": [1]}')
    manifest = tmp_path / "fold1.csv"
    manifest.write_bytes(b"image_id,split\na,train\n")

    evidence = build_dataset_evidence(dataset_root, manifest)

    # The DOI literal lives in the approved attribution paths only; the test
    # asserts the evidence carries the repository's recorded value.
    assert evidence["doi"] == DATASET_DOI
    assert evidence["version"] == DATASET_VERSION
    assert evidence["annotation_sha256"] == hashlib.sha256(b'{"images": [1]}').hexdigest()
    assert evidence["split_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert evidence["redistributed"] is False


def test_dataset_evidence_rejects_a_missing_manifest(tmp_path: Path):
    dataset_root = _official_dataset(tmp_path / "download")

    with pytest.raises(ValueError, match="split manifest does not exist"):
        build_dataset_evidence(dataset_root, tmp_path / "absent.csv")


def test_pretrained_evidence_records_the_real_digest_not_a_registry_value(tmp_path: Path):
    (tmp_path / "yolov8n.pt").write_bytes(BASE_BYTES)

    evidence = build_pretrained_evidence("yolov8n", _registry("yolov8n.pt"), tmp_path)

    assert evidence["sha256"] == BASE_DIGEST
    assert evidence["source_url"] == "https://example.invalid/yolov8n.pt"
    assert evidence["size_bytes"] == len(BASE_BYTES)
    # A cache path is machine-specific and must not travel with the evidence.
    assert "local_path" not in evidence


def test_missing_base_artifact_names_the_official_source(tmp_path: Path):
    with pytest.raises(ValueError, match="https://example.invalid/yolov8n.pt"):
        resolve_pretrained_artifact("yolov8n", _registry("yolov8n.pt"), tmp_path)


def test_digest_prefix_in_the_filename_is_checked(tmp_path: Path):
    # Torch Hub encodes the leading digest bytes in the filename; a file whose
    # content does not match it is the wrong artifact.
    (tmp_path / "model_coco-deadbeef.pth").write_bytes(BASE_BYTES)

    with pytest.raises(ValueError, match="deadbeef"):
        build_pretrained_evidence("yolov8n", _registry("model_coco-deadbeef.pth"), tmp_path)


def test_matching_digest_prefix_is_accepted(tmp_path: Path):
    name = f"model_coco-{BASE_DIGEST[:8]}.pth"
    (tmp_path / name).write_bytes(BASE_BYTES)

    evidence = build_pretrained_evidence("yolov8n", _registry(name), tmp_path)

    assert evidence["sha256"] == BASE_DIGEST


def test_evaluation_evidence_collects_settings_and_deviations(tmp_path: Path):
    results = _results(tmp_path / "results")

    evidence = build_evaluation_evidence(results, ("test",))

    assert evidence["detection_backends"] == ["cabbage_detection.ap_101"]
    assert evidence["splits"]["test"]["image_count"] == 92
    assert evidence["splits"]["test"]["detection"]["settings"]["ap_iou_threshold"] == 0.5
    assert evidence["splits"]["test"]["counting"]["settings"]["confidence_threshold"] == 0.5
    assert evidence["deviations"] == [
        "native detector score threshold lowered for full-range AP"
    ]


def test_evaluation_evidence_requires_the_summary(tmp_path: Path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="evaluation summary does not exist"):
        build_evaluation_evidence(tmp_path / "empty", ("test",))


def test_staging_copies_evidence_into_the_run_without_touching_the_source(tmp_path: Path):
    results = _results(tmp_path / "results")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    predictions, metrics = stage_run_evidence(run_dir, results, "test")

    assert predictions.parent == run_dir / "evaluation"
    assert predictions.read_text(encoding="utf-8") == "[]"
    assert metrics.read_bytes() == (results / "summary.json").read_bytes()
    assert (results / "test" / "records" / "predictions.json").is_file()


def test_staging_reports_missing_evaluation_output(tmp_path: Path):
    results = _results(tmp_path / "results")
    (results / "test" / "records" / "predictions.json").unlink()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="evaluation evidence is missing"):
        stage_run_evidence(run_dir, results, "test")


def test_repository_registry_covers_every_supported_model():
    from cabbage_detection.release_artifacts import MODEL_NAMES

    registry = read_weights_registry(Path("."))

    assert set(registry) == set(MODEL_NAMES)
    for name, record in registry.items():
        pretrained = record["pretrained"]
        for field in ("identifier", "expected_filename", "source_url"):
            assert str(pretrained.get(field, "")).strip(), f"{name}.{field}"


def test_registry_file_stays_parseable_yaml():
    payload = yaml.safe_load(Path("weights/provenance.yaml").read_text(encoding="utf-8"))
    assert isinstance(payload["models"], dict)
