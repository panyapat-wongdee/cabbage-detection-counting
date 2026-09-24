"""Run metadata serialization."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

import yaml

from .config import ExperimentConfig
from .artifact_paths import run_artifact_path


RESULT_STATUSES = {"smoke", "reproduction_candidate", "reproduced_verified"}
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _nested(metadata: Mapping[str, object], *keys: str) -> object | None:
    value: object = metadata
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _sha256_file(path: Path) -> str:
    """Hash a file in bounded-size chunks so large checkpoints stay streamable."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reproduction_evidence(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return sorted missing/invalid evidence fields; empty means promotable."""

    findings: set[str] = set()
    if metadata.get("result_status") != "reproduced_verified":
        findings.add("result_status")
    if metadata.get("status") != "succeeded":
        findings.add("status")
    if not isinstance(metadata.get("command"), list) or not metadata["command"]:
        findings.add("command")
    revision = _nested(metadata, "git", "revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        findings.add("git.revision")
    if not isinstance(_nested(metadata, "git", "dirty"), bool):
        findings.add("git.dirty")
    if not isinstance(metadata.get("config"), Mapping):
        findings.add("config")
    for field in ("doi", "version", "split_manifest_sha256", "annotation_sha256"):
        value = _nested(metadata, "dataset", field)
        if not isinstance(value, str) or not value:
            findings.add(f"dataset.{field}")
        elif field.endswith("sha256") and not _valid_digest(value):
            findings.add(f"dataset.{field}")
    for field in ("identifier", "source_url", "sha256"):
        value = _nested(metadata, "pretrained", field)
        if not isinstance(value, str) or not value:
            findings.add(f"pretrained.{field}")
        elif field == "sha256" and not _valid_digest(value):
            findings.add("pretrained.sha256")
    if not isinstance(metadata.get("environment"), Mapping) or not metadata["environment"]:
        findings.add("environment")
    if not isinstance(metadata.get("evaluation"), Mapping) or not metadata["evaluation"]:
        findings.add("evaluation")
    if not isinstance(metadata.get("deviations"), list):
        findings.add("deviations")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, Mapping):
        outputs = {}
    for name in ("predictions", "metrics", "best_checkpoint"):
        output = outputs.get(name)
        if not isinstance(output, Mapping) or not isinstance(output.get("path"), str) or not output["path"]:
            findings.add(f"outputs.{name}")
        elif not _valid_digest(output.get("sha256")):
            findings.add(f"outputs.{name}.sha256")
    return tuple(sorted(findings))


def verify_reproduction_run(run_dir: Path) -> tuple[str, ...]:
    """Verify output files still match the hashes recorded in run metadata."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("metadata",)
    outputs = metadata.get("outputs") if isinstance(metadata, Mapping) else None
    if not isinstance(outputs, Mapping):
        return ("outputs",)
    return _verify_output_descriptions(run_dir, outputs)


def _verify_output_descriptions(run_dir: Path, outputs: Mapping[str, object]) -> tuple[str, ...]:
    """Verify a run-relative output mapping before it is persisted as evidence."""

    findings: set[str] = set()
    for name, raw in outputs.items():
        if not isinstance(raw, Mapping):
            findings.add(f"outputs.{name}")
            continue
        relative = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            findings.add(f"outputs.{name}")
            continue
        try:
            path = (run_dir / relative).resolve()
            path.relative_to(run_dir.resolve())
            actual = _sha256_file(path)
        except (OSError, ValueError):
            findings.add(f"outputs.{name}")
            continue
        if actual != digest:
            findings.add(f"outputs.{name}.sha256")
    return tuple(sorted(findings))


def finalize_reproduction_run(
    run_dir: Path,
    predictions: Path,
    metrics: Path,
    *,
    dataset: Mapping[str, object],
    pretrained: Mapping[str, object],
    evaluation: Mapping[str, object],
    deviations: Sequence[str] = (),
) -> Path:
    """Attach evaluation evidence and promote a successful candidate run."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read run metadata: {metadata_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("run metadata must be an object")
    if payload.get("status") != "succeeded":
        raise ValueError("only succeeded runs can be promoted")
    if payload.get("result_status") != "reproduction_candidate":
        raise ValueError("only reproduction_candidate runs can be promoted")
    for label, value in (("dataset", dataset), ("pretrained", pretrained), ("evaluation", evaluation)):
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} evidence must be a mapping")
    described_outputs = payload.get("outputs", {})
    if not isinstance(described_outputs, Mapping):
        raise ValueError("run metadata outputs must be a mapping")
    described_outputs = dict(described_outputs)
    described_outputs["predictions"] = _describe_output(run_dir, Path(predictions))
    described_outputs["metrics"] = _describe_output(run_dir, Path(metrics))
    promoted = dict(payload)
    promoted.update(
        {
            "dataset": dict(dataset),
            "pretrained": dict(pretrained),
            "evaluation": dict(evaluation),
            "deviations": [str(item) for item in deviations],
            "outputs": described_outputs,
            "result_status": "reproduced_verified",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    output_findings = _verify_output_descriptions(run_dir, described_outputs)
    if output_findings:
        raise ValueError("run cannot be verified; output evidence is invalid: " + ", ".join(output_findings))
    findings = validate_reproduction_evidence(promoted)
    if findings:
        raise ValueError("run cannot be verified; missing evidence: " + ", ".join(findings))
    _atomic_json(metadata_path, promoted)
    return metadata_path


def _git_revision() -> str:
    try:
        repository = str(Path.cwd().resolve()).replace("\\", "/")
        return subprocess.check_output(
            ["git", f"-c", f"safe.directory={repository}", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_metadata() -> dict[str, object]:
    """Capture revision and dirty state without failing a run outside Git."""
    try:
        revision = _git_revision()
        dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-c",
                    f"safe.directory={Path.cwd().resolve().as_posix()}",
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON through a sibling temporary file and atomically replace it."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _describe_output(run_dir: Path, path: Path) -> dict[str, object]:
    resolved_run = run_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_run)
    except ValueError as error:
        raise ValueError(f"output must be inside run directory: {path}") from error
    if not resolved_path.is_file():
        raise FileNotFoundError(path)
    size = resolved_path.stat().st_size
    description: dict[str, object] = {"path": relative.as_posix(), "size_bytes": size}
    description["sha256"] = _sha256_file(resolved_path)
    return description


def _input_path_label(resolved: Path) -> str:
    """Return a portable label for an input path.

    An absolute path on the training machine carries the operator's home
    directory into run metadata that is published, so an input inside the
    repository is recorded relative to it. The digest below identifies the bytes
    either way.
    """
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.name


def _describe_input(path: Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(path)
    description: dict[str, object] = {"path": _input_path_label(resolved)}
    if resolved.is_dir():
        description["kind"] = "directory"
        description["hash_omitted_reason"] = "directory input is recorded by path; hash its manifest instead"
        return description
    description["kind"] = "file"
    description["size_bytes"] = resolved.stat().st_size
    if description["size_bytes"] > 50 * 1024 * 1024:
        description["hash_omitted_reason"] = "input exceeds 50 MiB release hash limit"
    else:
        description["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return description


@dataclass
class RunContext:
    """Lifecycle state for an immutable experiment run directory."""

    output_dir: Path
    metadata_path: Path
    _payload: dict[str, object]
    _finalized: bool = False

    def _write_running(self) -> None:
        if self._finalized:
            raise RuntimeError("run has already been finalized")
        _atomic_json(self.metadata_path, self._payload)

    def record_input(self, name: str, path: Path) -> None:
        """Record an input path and digest while the run is still active."""
        if not name or name in {"status", "outputs", "error"}:
            raise ValueError("input name is reserved or empty")
        inputs = dict(self._payload.get("inputs", {}))
        inputs[str(name)] = _describe_input(Path(path))
        self._payload["inputs"] = inputs
        self._write_running()

    def record_environment(self) -> dict[str, object]:
        """Capture available package and CUDA metadata without requiring them."""
        package_names = ("cabbage-detection", "torch", "torchvision", "ultralytics", "albumentations", "pycocotools", "PyYAML")
        packages: dict[str, str] = {}
        for package in package_names:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "not-installed"
        environment: dict[str, object] = {"python_version": sys.version, "packages": packages}
        try:
            import torch

            environment["cuda"] = {
                "available": bool(torch.cuda.is_available()),
                "count": int(torch.cuda.device_count()),
                "version": torch.version.cuda,
            }
        except ImportError:
            environment["cuda"] = {"available": False, "count": 0, "version": None}
        self._payload["environment"] = environment
        self._write_running()
        return environment

    def record_reproduction_context(
        self,
        *,
        dataset: Mapping[str, object],
        pretrained: Mapping[str, object],
        evaluation: Mapping[str, object],
        deviations: Sequence[str] = (),
    ) -> None:
        """Record result-sensitive provenance before finalizing a run."""
        if self._finalized:
            raise RuntimeError("run has already been finalized")
        if not isinstance(dataset, Mapping) or not isinstance(pretrained, Mapping) or not isinstance(evaluation, Mapping):
            raise TypeError("dataset, pretrained, and evaluation must be mappings")
        self._payload["dataset"] = dict(dataset)
        self._payload["pretrained"] = dict(pretrained)
        self._payload["evaluation"] = dict(evaluation)
        self._payload["deviations"] = [str(item) for item in deviations]
        self._write_running()

    def fail(self, error: BaseException) -> None:
        """Finalize a failed run with a type/message pair and no secret trace."""
        self._payload["error"] = {"type": type(error).__name__, "message": str(error)}
        self.finalize("failed", {})

    def finalize(
        self,
        status: Literal["succeeded", "failed"],
        outputs: Mapping[str, Path],
        *,
        result_status: Literal["smoke", "reproduction_candidate", "reproduced_verified"] = "reproduction_candidate",
        deviations: Sequence[str] | None = None,
    ) -> None:
        """Atomically finalize metadata while retaining failure evidence."""
        if status not in {"succeeded", "failed"}:
            raise ValueError("status must be succeeded or failed")
        if result_status not in RESULT_STATUSES:
            raise ValueError(f"result_status must be one of: {', '.join(sorted(RESULT_STATUSES))}")
        if self._finalized:
            raise RuntimeError("run has already been finalized")
        described = {
            str(name): _describe_output(self.output_dir, Path(path))
            for name, path in outputs.items()
        }
        payload = dict(self._payload)
        payload.update(
            {
                "status": status,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "outputs": described,
                "result_status": result_status,
                "deviations": list(deviations) if deviations is not None else list(self._payload.get("deviations", [])),
            }
        )
        if result_status == "reproduced_verified":
            findings = validate_reproduction_evidence(payload)
            if findings:
                raise ValueError("run cannot be verified; missing evidence: " + ", ".join(findings))
        _atomic_json(self.metadata_path, payload)
        self._payload = payload
        self._finalized = True


def create_run(config: ExperimentConfig, command: Sequence[str]) -> RunContext:
    """Create a new run directory and write its initial metadata.

    Existing directories are rejected so a run can never silently overwrite a
    prior experiment. Call :meth:`RunContext.finalize` exactly once.
    """
    output_dir = Path(config.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    config_path = write_resolved_config(config, output_dir)
    payload: dict[str, object] = {
        "status": "running",
        "result_status": "reproduction_candidate",
        "deviations": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "git": _git_metadata(),
        "python_version": sys.version,
        "config": config.to_dict(),
        "config_resolved": {"path": config_path.name, "sha256": _sha256_file(config_path)},
        "config_sha256": _sha256_file(config_path),
    }
    metadata_path = output_dir / "metadata.json"
    _atomic_json(metadata_path, payload)
    return RunContext(output_dir, metadata_path, payload)


def write_resolved_config(config: ExperimentConfig, run_dir: Path) -> Path:
    """Write the fully resolved run configuration as stable UTF-8 YAML."""

    destination = run_artifact_path(Path(run_dir), "config")
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        config.to_dict(),
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    destination.write_text(text, encoding="utf-8")
    return destination


def write_run_metadata(
    output_dir: Path,
    config: ExperimentConfig | None,
    command: Sequence[str] | None = None,
    *,
    filename: str = "run_metadata.json",
) -> Path:
    """Write a self-describing run metadata file and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not filename or Path(filename).name != filename:
        raise ValueError("metadata filename must be a simple file name")
    path = output_dir / filename
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command or sys.argv),
        "git_revision": _git_revision(),
        "python_version": sys.version,
        "config": config.to_dict() if config is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
