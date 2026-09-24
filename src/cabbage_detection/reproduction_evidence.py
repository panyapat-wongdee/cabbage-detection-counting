"""Assemble the evidence that promotes a reproduction candidate.

``finalize_reproduction_run`` deliberately takes explicit evidence documents
rather than inferring them, so that what a promotion asserts is reviewable. This
module builds those documents from artifacts that already exist — the weights
registry, the official dataset download, the run's split manifest, and the
evaluation output — and stages the prediction and metric files the promotion
points at inside the run directory.

Nothing here invents a value. Every field is read from a file, and a missing
input raises instead of falling back to a default, because a promotion that
rests on a guessed digest is worse than no promotion.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .data.official_layout import discover_official_dataset
from .release_artifacts import DATASET_DOI, MODEL_NAMES, TORCHVISION_MODELS

# The downloadable Mendeley record version this repository targets; the trailing
# component of DATASET_DOI names the same version.
DATASET_VERSION = "2"
PROVENANCE_PATH = Path("weights") / "provenance.yaml"
# Evidence staged inside a run directory, relative to it.
EVIDENCE_DIRNAME = "evaluation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evaluation artifact: {path}") from error


def read_weights_registry(root: Path = Path(".")) -> Mapping[str, Any]:
    """Return the ``models`` mapping of the weights provenance registry."""
    path = Path(root) / PROVENANCE_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read weights registry: {path}") from error
    models = (payload or {}).get("models")
    if not isinstance(models, Mapping):
        raise ValueError(f"weights registry has no models mapping: {path}")
    return models


def pretrained_search_paths(model_name: str, root: Path = Path(".")) -> tuple[Path, ...]:
    """Return the directories a base artifact is looked for, most likely first.

    torchvision downloads into the Torch Hub cache; the Ultralytics profiles
    resolve their base weights from the working directory. Both locations are
    reported in the error when the artifact is absent, so the operator knows
    where to put the file named by the registry.
    """
    root = Path(root)
    if model_name in TORCHVISION_MODELS:
        try:
            import torch

            hub = Path(torch.hub.get_dir()) / "checkpoints"
        except ImportError:
            hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        return (hub,)
    return (root, root / "weights")


def resolve_pretrained_artifact(
    model_name: str, registry: Mapping[str, Any], root: Path = Path(".")
) -> Path:
    """Locate the base weight file the registry names for ``model_name``."""
    record = registry.get(model_name)
    if not isinstance(record, Mapping):
        raise ValueError(f"weights registry has no entry for {model_name!r}")
    pretrained = record.get("pretrained")
    if not isinstance(pretrained, Mapping):
        raise ValueError(f"weights registry entry for {model_name!r} has no pretrained block")
    filename = str(pretrained.get("expected_filename", "")).strip()
    if not filename:
        raise ValueError(f"weights registry does not name a file for {model_name!r}")
    searched = pretrained_search_paths(model_name, root)
    for directory in searched:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    locations = ", ".join(str(directory) for directory in searched)
    raise ValueError(
        f"base artifact {filename} for {model_name} was not found in {locations}; "
        f"download it from {pretrained.get('source_url')} before promoting the run"
    )


def build_pretrained_evidence(
    model_name: str, registry: Mapping[str, Any], root: Path = Path(".")
) -> dict[str, Any]:
    """Describe the base artifact the run started from, with its real digest."""
    record = registry[model_name]
    pretrained = record["pretrained"]
    artifact = resolve_pretrained_artifact(model_name, registry, root)
    digest = _sha256(artifact)
    filename = str(pretrained["expected_filename"])
    # Torch Hub encodes the leading digest bytes in the filename; when it is
    # present it is a free integrity check on the located file.
    stem_digest = filename.rsplit("-", 1)[-1].split(".")[0]
    if len(stem_digest) == 8 and all(c in "0123456789abcdef" for c in stem_digest):
        if not digest.startswith(stem_digest):
            raise ValueError(
                f"{artifact} does not match the digest encoded in its name ({stem_digest})"
            )
    return {
        "identifier": str(pretrained["identifier"]),
        "source_url": str(pretrained["source_url"]),
        "sha256": digest,
        "expected_filename": filename,
        # The local cache path is deliberately not recorded: it is
        # machine-specific and the digest already identifies the bytes.
        "size_bytes": artifact.stat().st_size,
        "software_license": str(pretrained.get("software_license", "")),
        "training_dataset": str(pretrained.get("training_dataset", "")),
        "redistributed": bool(pretrained.get("redistributed", False)),
        "framework": str(record.get("framework", "")),
        "framework_version": str(record.get("framework_version", "")),
    }


def build_dataset_evidence(dataset_root: Path, split_manifest: Path) -> dict[str, Any]:
    """Describe the external dataset download and the split membership used."""
    layout = discover_official_dataset(Path(dataset_root))
    manifest = Path(split_manifest)
    if not manifest.is_file():
        raise ValueError(f"split manifest does not exist: {manifest}")
    return {
        "doi": DATASET_DOI,
        "version": DATASET_VERSION,
        "annotation_sha256": _sha256(layout.annotations_path),
        "split_manifest_sha256": _sha256(manifest),
        "split_manifest": manifest.as_posix(),
        "license": "CC BY 4.0",
        "redistributed": False,
    }


def build_evaluation_evidence(results_dir: Path, splits: Sequence[str]) -> dict[str, Any]:
    """Describe the scoring that the promotion rests on, split by split."""
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"evaluation summary does not exist: {summary_path}")
    summary = _read_json(summary_path)
    payload: dict[str, Any] = {
        "results_dir": results_dir.as_posix(),
        "summary_sha256": _sha256(summary_path),
        "detection_backends": summary.get("detection_backends", []),
        "splits": {},
    }
    deviations: set[str] = set()
    for split in splits:
        split_dir = results_dir / split
        detection = _read_json(split_dir / "detection.json")
        counting = _read_json(split_dir / "counting.json")
        metadata = _read_json(split_dir / "metadata.json")
        deviations.update(str(item) for item in metadata.get("deviations", ()))
        payload["splits"][split] = {
            "image_count": detection["scope"]["image_count"],
            "detection": {**detection["metrics"], "settings": detection["evaluation"]},
            "counting": {**counting["metrics"], "settings": counting["evaluation"]},
            "records": metadata.get("inputs", {}),
            "command": metadata.get("command", []),
            "git_revision": metadata.get("git_revision"),
        }
    payload["deviations"] = sorted(deviations)
    return payload


def stage_run_evidence(run_dir: Path, results_dir: Path, split: str) -> tuple[Path, Path]:
    """Copy the prediction and metric files a promotion points at into the run.

    ``finalize_reproduction_run`` requires its outputs to live inside the run
    directory so the evidence travels with the run. The originals under
    ``results/`` are left untouched and the copies are verified byte for byte.
    """
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    predictions_source = results_dir / split / "records" / "predictions.json"
    metrics_source = results_dir / "summary.json"
    for source in (predictions_source, metrics_source):
        if not source.is_file():
            raise ValueError(f"evaluation evidence is missing: {source}")
    destination_dir = run_dir / EVIDENCE_DIRNAME
    destination_dir.mkdir(parents=True, exist_ok=True)
    predictions = destination_dir / f"{split}_predictions.json"
    metrics = destination_dir / "metrics.json"
    for source, destination in ((predictions_source, predictions), (metrics_source, metrics)):
        shutil.copyfile(source, destination)
        if _sha256(source) != _sha256(destination):
            raise ValueError(f"staged evidence does not match its source: {source}")
    return predictions, metrics


@dataclass(frozen=True)
class CollectedEvidence:
    """Paths and deviations that ``finalize-reproduction`` consumes verbatim."""

    model_name: str
    dataset: Path
    pretrained: Path
    evaluation: Path
    predictions: Path
    metrics: Path
    deviations: tuple[str, ...]


def collect_run_evidence(
    run_dir: Path,
    results_dir: Path,
    dataset_root: Path,
    *,
    split: str = "test",
    splits: Sequence[str] = ("train", "val", "test"),
    root: Path = Path("."),
) -> CollectedEvidence:
    """Write the three evidence documents and stage the promotion outputs.

    Returns the paths ``finalize-reproduction`` needs, so the promotion command
    repeats no value that was derived here.
    """
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    config_path = run_dir / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read run config: {config_path}") from error
    model_name = str((config.get("model") or {}).get("name", ""))
    if model_name not in MODEL_NAMES:
        raise ValueError(f"run config names an unsupported model: {model_name!r}")
    manifest = Path(str((config.get("dataset") or {}).get("split_manifest", "")))

    registry = read_weights_registry(root)
    evidence_dir = run_dir / EVIDENCE_DIRNAME
    evidence_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "dataset": build_dataset_evidence(dataset_root, manifest),
        "pretrained": build_pretrained_evidence(model_name, registry, root),
        "evaluation": build_evaluation_evidence(results_dir, splits),
    }
    written: dict[str, Path] = {}
    for name, payload in documents.items():
        path = evidence_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        written[name] = path
    predictions, metrics = stage_run_evidence(run_dir, results_dir, split)
    return CollectedEvidence(
        model_name=model_name,
        dataset=written["dataset"],
        pretrained=written["pretrained"],
        evaluation=written["evaluation"],
        predictions=predictions,
        metrics=metrics,
        deviations=tuple(documents["evaluation"]["deviations"]),
    )
