from dataclasses import replace
import json
from pathlib import Path

import pytest

from cabbage_detection.release_artifacts import (
    ReleaseValidationError,
    load_release_manifest,
    validate_release_manifest,
    verify_release_sources,
)


MANIFEST = Path("tests/release/fixture-manifest.json")


def test_manifest_accepts_verified_best_checkpoint() -> None:
    manifest = load_release_manifest(MANIFEST)
    assert manifest.release_tag == "reproduced-checkpoints-v1.0.0"
    assert len(manifest.checkpoints) == 1
    assert manifest.checkpoints[0].result_status == "reproduced_verified"
    assert Path(manifest.checkpoints[0].source_path).name in {"best.pt", "best.pth"}


def test_manifest_rejects_non_best_source() -> None:
    manifest = load_release_manifest(MANIFEST)
    record = manifest.checkpoints[0]
    with pytest.raises(ReleaseValidationError, match="not a best checkpoint"):
        validate_release_manifest(
            manifest.with_checkpoints((replace(record, source_path=record.source_path.replace("best", "last")),))
        )


def test_manifest_rejects_unverified_run() -> None:
    manifest = load_release_manifest(MANIFEST)
    record = manifest.checkpoints[0]
    with pytest.raises(ReleaseValidationError, match="reproduced_verified"):
        validate_release_manifest(
            manifest.with_checkpoints((replace(record, result_status="smoke"),))
        )


def test_hash_mismatch_fails_closed(release_fixture) -> None:
    root, manifest, _ = release_fixture
    source = root / manifest.checkpoints[0].source_path
    source.write_bytes(b"changed")
    with pytest.raises(ReleaseValidationError, match="SHA-256 mismatch|size mismatch"):
        verify_release_sources(root, manifest)


def test_manifest_rejects_schema_version(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "release_tag": "reproduced-checkpoints-v1.0.0",
        "dataset_doi": "10.17632/" + "5cp2dyjczk.2",
        "checkpoints": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="unknown manifest key"):
        load_release_manifest(path)


def test_release_source_must_match_metadata_best_checkpoint(release_fixture) -> None:
    root, manifest, _ = release_fixture
    metadata_path = root / manifest.checkpoints[0].metadata_path
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["outputs"] = {"best_checkpoint": {"path": "checkpoints/other.pt"}}
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="best checkpoint path"):
        verify_release_sources(root, manifest)
