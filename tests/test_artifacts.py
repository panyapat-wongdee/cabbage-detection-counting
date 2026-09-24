import json
from pathlib import Path

import pytest
import yaml

from cabbage_detection.artifacts import (
    create_run,
    finalize_reproduction_run,
    validate_reproduction_evidence,
    verify_reproduction_run,
)
from cabbage_detection.artifact_paths import run_artifact_path
from cabbage_detection.config import AugmentationConfig, DatasetConfig, EvaluationConfig, ExperimentConfig, InitializationConfig, NativeTrainingConfig, SchedulerConfig


def _config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="faster_rcnn",
        framework="torchvision",
        dataset=DatasetConfig("pascal_voc", Path("data/fold1_pascal"), Path("splits/fold1_recovered.csv"), None, "labels", ("background", "cabbage")),
        initialization=InitializationConfig("COCO_V1", 2),
        image_size=(512, 512),
        optimizer="SGD",
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        batch_size=16,
        epochs=1,
        augmentation=AugmentationConfig("reproduction_augmented", 0.5, 0.5, 0.1, 0.1, 15.0, 0.1, 0.05, 0.05, 0.0, True, 114),
        scheduler=SchedulerConfig("step_lr", 1000, 0.1, None),
        evaluation=EvaluationConfig(100, 1.0, "repository", "torchvision"),
        native_training=NativeTrainingConfig(500, False, True, False),
        confidence_threshold=0.5,
        iou_threshold=0.5,
        output_dir=output_dir,
    )


def test_create_run_refuses_existing_directory(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        create_run(_config(output), ["train"])


def test_run_context_writes_atomic_metadata_and_hashes_outputs(tmp_path: Path):
    output = tmp_path / "run"
    context = create_run(_config(output), ["train", "--config", "x.yaml"])
    artifact = output / "metrics.json"
    artifact.write_text('{"map50": 1.0}\n', encoding="utf-8")

    context.finalize("succeeded", {"metrics": artifact})

    payload = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["command"] == ["train", "--config", "x.yaml"]
    assert payload["config"]["model"]["name"] == "faster_rcnn"
    assert payload["outputs"]["metrics"]["path"] == "metrics.json"
    assert len(payload["outputs"]["metrics"]["sha256"]) == 64
    assert payload["git"]["revision"]
    assert "finished_at_utc" in payload
    resolved = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
    assert resolved == payload["config"]
    assert not (output / "config.resolved.json").exists()
    assert payload["config_resolved"]["path"] == "config.yaml"
    assert len(payload["config_sha256"]) == 64


def test_run_artifact_paths_are_grouped_and_explicit(tmp_path: Path):
    run = tmp_path / "run"
    assert run_artifact_path(run, "config") == run / "config.yaml"
    assert run_artifact_path(run, "console_log") == run / "logs" / "console.log"
    assert run_artifact_path(run, "training_log") == run / "logs" / "training_log.csv"
    assert run_artifact_path(run, "best_epoch") == run / "logs" / "best_epoch.json"
    assert run_artifact_path(run, "best_checkpoint") == run / "checkpoints" / "best.pt"
    assert run_artifact_path(run, "last_checkpoint") == run / "checkpoints" / "last.pt"
    with pytest.raises(KeyError):
        run_artifact_path(run, "unknown")


def test_run_context_streams_hash_for_large_output(tmp_path: Path):
    output = tmp_path / "run"
    context = create_run(_config(output), ["train"])
    artifact = output / "large.bin"
    with artifact.open("wb") as stream:
        stream.truncate(50 * 1024 * 1024 + 1)

    context.finalize("succeeded", {"large": artifact})

    payload = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(payload["outputs"]["large"]["sha256"]) == 64


def test_failed_run_retains_error_status(tmp_path: Path):
    context = create_run(_config(tmp_path / "run"), ["train"])
    context.finalize("failed", {})
    payload = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"


def test_run_context_records_inputs_and_environment(tmp_path: Path):
    context = create_run(_config(tmp_path / "run"), ["train"])
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,image_id\ntrain,a\n", encoding="utf-8")
    context.record_input("manifest", manifest)
    environment = context.record_environment()

    assert len(environment["python_version"]) > 0
    payload = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    # An absolute path would carry the operator's home directory into published
    # run metadata; an input outside the repository keeps its name only, and the
    # digest below identifies the bytes.
    assert payload["inputs"]["manifest"]["path"] == manifest.name
    assert len(payload["inputs"]["manifest"]["sha256"]) == 64
    assert "packages" in payload["environment"]


def test_run_context_fail_preserves_exception_details(tmp_path: Path):
    context = create_run(_config(tmp_path / "run"), ["train"])
    context.fail(RuntimeError("synthetic failure"))
    payload = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == {"type": "RuntimeError", "message": "synthetic failure"}


def test_verified_evidence_requires_complete_provenance() -> None:
    metadata = {
        "result_status": "reproduced_verified",
        "status": "succeeded",
        "command": ["train"],
        "git": {"revision": "a" * 40, "dirty": False},
        "config": {"model": {"name": "faster_rcnn"}},
        "dataset": {
            "doi": "10.17632/5cp2dyjczk.2",
            "version": "2",
            "split_manifest_sha256": "b" * 64,
            "annotation_sha256": "c" * 64,
        },
        "pretrained": {
            "identifier": "COCO_V1",
            "source_url": "https://example.test/weights.pth",
            "sha256": "d" * 64,
        },
        "environment": {"python_version": "3.11"},
        "evaluation": {"iou_threshold": 0.5},
        "deviations": [],
        "outputs": {
            "predictions": {"path": "predictions.json", "sha256": "e" * 64},
            "metrics": {"path": "metrics.json", "sha256": "f" * 64},
            "best_checkpoint": {"path": "best.pth", "sha256": "0" * 64},
        },
    }
    assert validate_reproduction_evidence(metadata) == ()


def test_verified_evidence_reports_missing_fields() -> None:
    findings = validate_reproduction_evidence(
        {"result_status": "reproduced_verified", "status": "succeeded"}
    )
    assert "git.revision" in findings
    assert "dataset.doi" in findings
    assert "pretrained.sha256" in findings
    assert "outputs.best_checkpoint" in findings


def _candidate_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    context = create_run(_config(run), ["train"])
    context.record_environment()
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"best")
    context.finalize("succeeded", {"best_checkpoint": checkpoint})
    predictions = run / "predictions.json"
    predictions.write_text('{"records": []}\n', encoding="utf-8")
    metrics = run / "metrics.json"
    metrics.write_text('{"map50": 0.5}\n', encoding="utf-8")
    return run, predictions, metrics


def _complete_evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        {
            "doi": "10.17632/5cp2dyjczk.2",
            "version": "2",
            "split_manifest_sha256": "b" * 64,
            "annotation_sha256": "c" * 64,
        },
        {"identifier": "COCO_V1", "source_url": "https://example.test/weights.pth", "sha256": "d" * 64},
        {"backend": "pycocotools.coco_eval", "iou_threshold": 0.5, "confidence_threshold": 0.5},
    )


def test_finalize_reproduction_run_promotes_complete_candidate(tmp_path: Path):
    run, predictions, metrics = _candidate_run(tmp_path)
    dataset, pretrained, evaluation = _complete_evidence()

    metadata_path = finalize_reproduction_run(
        run,
        predictions,
        metrics,
        dataset=dataset,
        pretrained=pretrained,
        evaluation=evaluation,
        deviations=(),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["result_status"] == "reproduced_verified"
    assert metadata["outputs"]["predictions"]["path"] == "predictions.json"
    assert metadata["outputs"]["metrics"]["path"] == "metrics.json"
    assert verify_reproduction_run(run) == ()


def test_finalize_reproduction_run_rejects_incomplete_evidence(tmp_path: Path):
    run, predictions, metrics = _candidate_run(tmp_path)

    with pytest.raises(ValueError, match="dataset.annotation_sha256"):
        finalize_reproduction_run(
            run,
            predictions,
            metrics,
            dataset={},
            pretrained={},
            evaluation={},
            deviations=(),
        )


def test_finalize_reproduction_run_rejects_smoke_run(tmp_path: Path):
    run, predictions, metrics = _candidate_run(tmp_path)
    metadata_path = run / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["result_status"] = "smoke"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dataset, pretrained, evaluation = _complete_evidence()

    with pytest.raises(ValueError, match="only reproduction_candidate runs can be promoted"):
        finalize_reproduction_run(
            run,
            predictions,
            metrics,
            dataset=dataset,
            pretrained=pretrained,
            evaluation=evaluation,
        )


def test_finalize_reproduction_run_rejects_tampered_existing_output(tmp_path: Path):
    run, predictions, metrics = _candidate_run(tmp_path)
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"tampered")
    dataset, pretrained, evaluation = _complete_evidence()

    with pytest.raises(ValueError, match="outputs.best_checkpoint.sha256"):
        finalize_reproduction_run(
            run,
            predictions,
            metrics,
            dataset=dataset,
            pretrained=pretrained,
            evaluation=evaluation,
            deviations=(),
        )

    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["result_status"] == "reproduction_candidate"


def test_verify_reproduction_run_detects_tampered_output(tmp_path: Path):
    run, predictions, metrics = _candidate_run(tmp_path)
    dataset, pretrained, evaluation = _complete_evidence()
    finalize_reproduction_run(
        run,
        predictions,
        metrics,
        dataset=dataset,
        pretrained=pretrained,
        evaluation=evaluation,
        deviations=(),
    )
    metrics.write_text('{"map50": 0.9}\n', encoding="utf-8")

    findings = verify_reproduction_run(run)

    assert "outputs.metrics.sha256" in findings
