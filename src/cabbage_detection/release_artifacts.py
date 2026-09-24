"""Safe organization and packaging of verified reproduced checkpoints.

The module deliberately treats model files as opaque bytes.  It hashes and
copies them, but never deserializes a checkpoint while preparing a release.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import yaml
from .progress import progress, status


DATASET_DOI = "10.17632/5cp2dyjczk.2"
RELEASE_TAG = "reproduced-checkpoints-v1.0.0"
# The tag names the artifact kind, not the repository version, so a later
# source release can be tagged independently. The title is kept beside it and
# in ASCII so it survives a shell argument on any platform.
RELEASE_TITLE = "Reproduced checkpoints v1.0.0: fourteen detection models and five input-size variants"
SOURCE_REPOSITORY_URL = "https://github.com/panyapat-wongdee/cabbage-detection-counting"
ULTRALYTICS_SOURCE_URL = "https://github.com/ultralytics/ultralytics"
# The nine architectures compared in the cited study.
PUBLISHED_MODELS = frozenset(
    {"faster_rcnn", "ssd", "retinanet", "fcos", "yolov8n", "yolov8m", "yolo11n", "yolo11m", "rt-detr-l"}
)
# Trained in the original work but not compared in the paper.
SUPPLEMENTARY_MODELS = frozenset({"ssdlite"})
# Added by this repository after the study; never part of the original work.
EXTENSION_MODELS = ("yolo12n", "yolo12m", "yolo26n", "yolo26m")
MODEL_NAMES = (
    "faster_rcnn",
    "ssd",
    "retinanet",
    "fcos",
    "yolov8n",
    "yolov8m",
    "yolo11n",
    "yolo11m",
    "rt-detr-l",
    "ssdlite",
    *EXTENSION_MODELS,
)
# reproduced-checkpoints-v1.0.0 holds every model, plus the named variant
# runs of some of them (``<model>-<variant>``, e.g. ``fcos-detector512``).
RELEASE_MODELS = frozenset(MODEL_NAMES)
RELEASE_VARIANTS = ("detector512",)
TORCHVISION_MODELS = frozenset({"faster_rcnn", "ssd", "retinanet", "fcos", "ssdlite"})
ULTRALYTICS_MODELS = frozenset(set(MODEL_NAMES) - TORCHVISION_MODELS)
CONDITIONS = ("augmented", "no_augmentation")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReleaseValidationError(ValueError):
    """Raised when an artifact or release manifest is unsafe to publish."""


@dataclass(frozen=True)
class CheckpointRecord:
    model_name: str
    framework: str
    study_scope: str
    condition: str
    run_id: str
    result_status: str
    source_path: str
    metadata_path: str
    size_bytes: int
    sha256: str
    fine_tuned_license: str
    base_identifier: str
    base_source_url: str
    base_artifact_sha256: str
    config_sha256: str
    git_commit: str


@dataclass(frozen=True)
class ReleaseManifest:
    release_tag: str
    dataset_doi: str
    checkpoints: tuple[CheckpointRecord, ...]

    def with_checkpoints(self, checkpoints: tuple[CheckpointRecord, ...]) -> "ReleaseManifest":
        return replace(self, checkpoints=checkpoints)


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory without interpreting its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_from_json(raw: Mapping[str, object]) -> CheckpointRecord:
    required = {"model_name", "framework", "study_scope", "condition", "run_id", "result_status", "source_path", "metadata_path", "size_bytes", "sha256", "fine_tuned_license", "base_identifier", "base_source_url", "base_artifact_sha256", "config_sha256", "git_commit"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ReleaseValidationError(f"manifest record missing fields: {', '.join(missing)}")
    unknown = sorted(set(raw) - required)
    if unknown:
        raise ReleaseValidationError(f"unknown manifest record key(s): {', '.join(unknown)}")
    return CheckpointRecord(
        model_name=str(raw["model_name"]), framework=str(raw["framework"]), study_scope=str(raw["study_scope"]),
        condition=str(raw["condition"]), run_id=str(raw["run_id"]), result_status=str(raw["result_status"]),
        source_path=str(raw["source_path"]), metadata_path=str(raw["metadata_path"]), size_bytes=int(raw["size_bytes"]), sha256=str(raw["sha256"]),
        fine_tuned_license=str(raw["fine_tuned_license"]), base_identifier=str(raw["base_identifier"]), base_source_url=str(raw["base_source_url"]),
        base_artifact_sha256=str(raw["base_artifact_sha256"]), config_sha256=str(raw["config_sha256"]), git_commit=str(raw["git_commit"]),
    )


def study_scope(model_name: str) -> str:
    """Return the publication scope label a model carries in every report."""
    if model_name in PUBLISHED_MODELS:
        return "paper_model"
    if model_name in SUPPLEMENTARY_MODELS:
        return "supplementary_unreported"
    if model_name in EXTENSION_MODELS:
        return "repository_extension"
    raise ValueError(f"unknown model: {model_name}")


def expected_release_attributes(model_name: str) -> tuple[str, str, str]:
    """Return the (study scope, framework, fine-tuned licence) a model must carry.

    Manifest generation and manifest validation read these from here so they
    cannot drift apart: a generated record is checked against the same rule that
    rejects a hand-written one.
    """
    torchvision = model_name in TORCHVISION_MODELS
    return (
        study_scope(model_name),
        "torchvision" if torchvision else "ultralytics",
        "Apache-2.0" if torchvision else "AGPL-3.0-only",
    )


def release_run_ids(model_name: str) -> tuple[str, ...]:
    """Run directory names a model may be released from: itself and its variants."""
    return (model_name, *(f"{model_name}-{variant}" for variant in RELEASE_VARIANTS))


def _validate_record(record: CheckpointRecord) -> None:
    if record.model_name not in RELEASE_MODELS or record.condition not in CONDITIONS:
        raise ReleaseValidationError("manifest contains an unsupported model or condition")
    if record.run_id not in release_run_ids(record.model_name):
        raise ReleaseValidationError(f"run {record.run_id!r} is not a released run of {record.model_name}")
    if Path(record.source_path).parts[2:3] != (record.run_id,):
        raise ReleaseValidationError(f"source path does not belong to run {record.run_id}: {record.source_path}")
    expected_scope, expected_framework, expected_license = expected_release_attributes(
        record.model_name
    )
    if record.study_scope != expected_scope or record.framework != expected_framework or record.fine_tuned_license != expected_license:
        raise ReleaseValidationError(f"invalid scope/framework/license for {record.model_name}")
    if record.result_status != "reproduced_verified":
        raise ReleaseValidationError(f"result_status must be reproduced_verified: {record.model_name}")
    source, metadata = Path(record.source_path), Path(record.metadata_path)
    if not record.source_path.startswith("runs/reproduced/") or "\\" in record.source_path:
        raise ReleaseValidationError(f"source path is not a portable reproduced path: {record.source_path}")
    if not record.metadata_path.startswith("runs/reproduced/") or "\\" in record.metadata_path:
        raise ReleaseValidationError(f"metadata path is not a portable reproduced path: {record.metadata_path}")
    if source.is_absolute() or metadata.is_absolute() or ".." in source.parts or ".." in metadata.parts:
        raise ReleaseValidationError("manifest path escapes the repository")
    if not metadata.parts[:-1] == source.parts[: len(metadata.parts) - 1]:
        raise ReleaseValidationError("source and metadata must belong to the same reproduced run")
    if source.name not in {"best.pt", "best.pth"}:
        raise ReleaseValidationError(f"release source is not a best checkpoint: {record.source_path}")
    if record.size_bytes <= 0 or not _SHA256_RE.fullmatch(record.sha256):
        raise ReleaseValidationError(f"invalid size or SHA-256 for {record.source_path}")
    if not _SHA256_RE.fullmatch(record.base_artifact_sha256) or not _SHA256_RE.fullmatch(record.config_sha256):
        raise ReleaseValidationError(f"invalid provenance digest for {record.model_name}")
    if not _GIT_RE.fullmatch(record.git_commit) or not record.base_identifier or not record.base_source_url.startswith("https://"):
        raise ReleaseValidationError(f"invalid provenance for {record.model_name}")


def validate_release_manifest(manifest: ReleaseManifest) -> None:
    """Validate the best-checkpoint matrix for new reproduction runs."""
    if manifest.release_tag != RELEASE_TAG:
        raise ReleaseValidationError("unsupported release manifest tag")
    if manifest.dataset_doi != DATASET_DOI:
        raise ReleaseValidationError("manifest dataset DOI does not match the approved dataset")
    actual = {(record.run_id, record.condition) for record in manifest.checkpoints}
    if not actual or len(actual) != len(manifest.checkpoints):
        raise ReleaseValidationError("manifest must contain at least one unique run/condition record")
    for record in manifest.checkpoints:
        _validate_record(record)


def describe_run_checkpoint(
    root: Path, run_dir: Path, condition: str = "augmented"
) -> dict[str, object]:
    """Build one manifest record from a promoted run's own evidence.

    Every field is copied from the run's immutable metadata and its collected
    base-weight evidence, so the manifest cannot state a digest that the run
    itself does not already record. The checkpoint is re-hashed rather than
    trusted, because the manifest is what a downloader checks against.
    """
    root, run_dir = Path(root), Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(f"cannot read run metadata: {metadata_path}") from error
    if metadata.get("result_status") != "reproduced_verified":
        raise ReleaseValidationError(
            f"{run_dir.name} is {metadata.get('result_status')!r}; promote it before releasing it"
        )
    pretrained_path = run_dir / "evaluation" / "pretrained.json"
    try:
        pretrained = json.loads(pretrained_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseValidationError(
            f"cannot read base-weight evidence: {pretrained_path}"
        ) from error

    model_name = str((metadata.get("config", {}).get("model") or {}).get("name", ""))
    if model_name not in MODEL_NAMES:
        raise ReleaseValidationError(f"run metadata names an unsupported model: {model_name!r}")
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ReleaseValidationError(f"best checkpoint is missing: {checkpoint}")
    recorded = (metadata.get("outputs", {}) or {}).get("best_checkpoint", {})
    digest = sha256_file(checkpoint)
    if recorded.get("sha256") != digest:
        raise ReleaseValidationError(
            f"{checkpoint} no longer matches the digest recorded by its run"
        )
    study_scope, framework, fine_tuned_license = expected_release_attributes(model_name)
    return {
        "model_name": model_name,
        "framework": framework,
        "study_scope": study_scope,
        "condition": condition,
        "run_id": run_dir.name,
        "result_status": str(metadata["result_status"]),
        "source_path": checkpoint.relative_to(root).as_posix(),
        "metadata_path": metadata_path.relative_to(root).as_posix(),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": digest,
        "fine_tuned_license": fine_tuned_license,
        "base_identifier": str(pretrained["identifier"]),
        "base_source_url": str(pretrained["source_url"]),
        "base_artifact_sha256": str(pretrained["sha256"]),
        "config_sha256": str(metadata["config_sha256"]),
        "git_commit": str(metadata["git"]["revision"]),
    }


def build_release_manifest(
    root: Path, run_dirs: Sequence[Path], condition: str = "augmented"
) -> dict[str, object]:
    """Assemble the release manifest for a set of promoted runs."""
    records = [describe_run_checkpoint(root, run_dir, condition) for run_dir in run_dirs]
    if not records:
        raise ReleaseValidationError("no promoted runs were given")
    payload = {
        "release_tag": RELEASE_TAG,
        "dataset_doi": DATASET_DOI,
        "checkpoints": sorted(
            records, key=lambda item: (MODEL_NAMES.index(item["model_name"]), item["run_id"])
        ),
    }
    # Fail here rather than at packaging time.
    validate_release_manifest(
        ReleaseManifest(
            release_tag=RELEASE_TAG,
            dataset_doi=DATASET_DOI,
            checkpoints=tuple(_record_from_json(item) for item in payload["checkpoints"]),
        )
    )
    return payload


def load_release_manifest(path: Path) -> ReleaseManifest:
    """Load and validate the current schema-less JSON manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseValidationError("release manifest must be a JSON object")
    required = {"release_tag", "dataset_doi", "checkpoints"}
    missing = sorted(required - set(payload))
    if missing:
        raise ReleaseValidationError(f"manifest missing key(s): {', '.join(missing)}")
    unknown = sorted(set(payload) - required)
    if unknown:
        raise ReleaseValidationError(f"unknown manifest key(s): {', '.join(unknown)}")
    if not isinstance(payload["checkpoints"], list):
        raise ReleaseValidationError("manifest checkpoints must be a list")
    manifest = ReleaseManifest(
        release_tag=str(payload["release_tag"]),
        dataset_doi=str(payload["dataset_doi"]),
        checkpoints=tuple(_record_from_json(item) for item in payload["checkpoints"]),
    )
    validate_release_manifest(manifest)
    return manifest


def verify_release_sources(root: Path, manifest: ReleaseManifest) -> None:
    """Verify every checkpoint and its immutable evidence metadata."""
    validate_release_manifest(manifest)
    failures: list[str] = []
    root = Path(root)
    for record in manifest.checkpoints:
        metadata_path = root / Path(record.metadata_path)
        if not metadata_path.is_file():
            failures.append(f"missing evidence metadata: {record.metadata_path}")
        else:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or metadata.get("result_status") != "reproduced_verified":
                    failures.append(f"evidence is not verified: {record.metadata_path}")
                outputs = metadata.get("outputs")
                best = outputs.get("best_checkpoint") if isinstance(outputs, dict) else None
                if not isinstance(best, dict) or not isinstance(best.get("path"), str):
                    failures.append(f"missing best checkpoint metadata: {record.model_name}")
                else:
                    best_path = Path(record.metadata_path).parent / Path(best["path"])
                    if best_path.as_posix() != record.source_path:
                        failures.append(f"best checkpoint path mismatch: {record.model_name}")
                    if best.get("sha256") != record.sha256:
                        failures.append(f"best checkpoint SHA-256 mismatch: {record.model_name}")
                    if best.get("size_bytes") != record.size_bytes:
                        failures.append(f"best checkpoint size mismatch: {record.model_name}")
            except (OSError, json.JSONDecodeError):
                failures.append(f"invalid evidence metadata: {record.metadata_path}")
        path = root / Path(record.source_path)
        if not path.is_file():
            failures.append(f"missing source: {record.source_path}")
            continue
        if path.stat().st_size != record.size_bytes:
            failures.append(f"size mismatch: {record.source_path}")
        elif sha256_file(path) != record.sha256:
            failures.append(f"SHA-256 mismatch: {record.source_path}")
    if failures:
        raise ReleaseValidationError("; ".join(failures))


def build_sidecar(record: CheckpointRecord, registry_path: Path) -> dict[str, object]:
    """Build portable JSON metadata for one reproduced checkpoint."""
    registry_path = Path(registry_path)
    if registry_path.is_file():
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        registry_record = registry["models"][record.model_name]
        redistributed = bool(registry_record["pretrained"]["redistributed"])
    else:
        redistributed = False
    return {
        "model_name": record.model_name,
        "framework": record.framework,
        "study_scope": record.study_scope,
        "run_id": record.run_id,
        "result_status": record.result_status,
        "condition": record.condition,
        "checkpoint": {"filename": Path(record.source_path).name, "size_bytes": record.size_bytes, "sha256": record.sha256},
        "base": {
            "identifier": record.base_identifier,
            "source_url": record.base_source_url,
            "sha256": record.base_artifact_sha256,
            "redistributed": redistributed,
        },
        "git_commit": record.git_commit,
        "config_sha256": record.config_sha256,
        "fine_tuned_license": record.fine_tuned_license,
        "dataset_doi": DATASET_DOI,
        "evidence": [record.metadata_path],
    }


def _launch_config(root: Path, record: CheckpointRecord) -> str:
    """The config file the run was launched with, read from its metadata."""
    metadata = json.loads((Path(root) / record.metadata_path).read_text(encoding="utf-8"))
    command = [str(part) for part in metadata.get("command", [])]
    if "--config" not in command or command.index("--config") + 1 >= len(command):
        raise ReleaseValidationError(f"run metadata records no --config: {record.metadata_path}")
    return command[command.index("--config") + 1].replace("\\", "/")


def render_model_card(run_id: str, records: tuple[CheckpointRecord, ...], root: Path) -> str:
    """Render a short model card without copying raw run metadata."""
    status = records[0].study_scope
    framework = records[0].framework
    license_name = records[0].fine_tuned_license
    label = {
        "paper_model": "study architecture",
        "supplementary_unreported": "supplementary experiment",
        "repository_extension": "repository extension model",
    }[status]
    conditions = " and ".join(f"`{record.condition}`" for record in records)
    config = _launch_config(root, records[0])
    variant = ""
    if run_id != records[0].model_name:
        variant = (
            f"This is the `{run_id.removeprefix(records[0].model_name + '-')}` variant of "
            f"`{records[0].model_name}`: its detector was trained on its configured input size, "
            "so it must be used with the config below, not with the model's primary config.\n\n"
        )
    return (
        f"# {run_id}\n\n"
        f"This archive contains best checkpoints from newly executed repository runs for this {label}.\n\n"
        f"{variant}"
        f"Framework: `{framework}`  \nFine-tuned asset license: `{license_name}`  \n"
        f"Config: `{config}`  \nConditions: {conditions}, 500 epochs.\n\n"
        "See CHECKPOINTS.json for byte sizes, SHA-256 digests, and run evidence.\n"
    )


# Notice files bundled beside LICENSE, keyed by archive member name under
# ``notices/``. Each is copied verbatim from the repository so an archive
# carries the text itself, not only a pointer to it.
_COMMON_NOTICES = {
    "MS-COCO.md": "weights/notices/MS-COCO.md",
    "cabbage-dataset.md": "weights/notices/cabbage-dataset.md",
}
TORCHVISION_NOTICES = {
    "BSD-3-Clause-torchvision.txt": "LICENSES/BSD-3-Clause-torchvision.txt",
    **_COMMON_NOTICES,
}
ULTRALYTICS_NOTICES = {"ultralytics.md": "weights/notices/ultralytics.md", **_COMMON_NOTICES}


def notice_sources(model_name: str) -> dict[str, str]:
    """Return ``{member name: repository path}`` for the model's bundled notices."""
    return dict(TORCHVISION_NOTICES if model_name in TORCHVISION_MODELS else ULTRALYTICS_NOTICES)


def render_third_party_notices(
    model_name: str, root: Path, git_commit: str | None = None, source_commit: str | None = None
) -> str:
    """Render the notices required by the model's framework and data lineage.

    ``source_commit`` is the published commit that serves as the corresponding
    source; ``git_commit`` is the revision the run recorded when it trained,
    kept as provenance. The two differ because the development history was
    consolidated into a single commit before publication.
    """
    if model_name in TORCHVISION_MODELS:
        return (
            "This fine-tuned checkpoint includes an author contribution licensed under Apache-2.0.\n\n"
            "The model was initialized from a torchvision COCO_V1 artifact. The torchvision "
            "BSD-3-Clause notice is retained in notices/BSD-3-Clause-torchvision.txt and the MS COCO "
            "provenance in notices/MS-COCO.md; the COCO image licenses are not relicensed here.\n\n"
            f"Training data attribution: external cabbage dataset, DOI {DATASET_DOI}, CC BY 4.0; "
            "see notices/cabbage-dataset.md.\n"
        )
    # AGPL-3.0-only obliges the distributor to offer the corresponding source,
    # so the notice names a commit that the public repository actually serves.
    if not source_commit or not _GIT_RE.fullmatch(source_commit):
        raise ReleaseValidationError(
            f"{model_name}: an Ultralytics archive needs the published source commit (40 hex digits)"
        )
    # The run's recorded revision predates the history consolidation; the
    # training and inference code for every released run is unchanged at the
    # published commit, which is therefore the corresponding source.
    trained = (
        f" The revision recorded in CHECKPOINTS.json (`{git_commit}`) belongs to the "
        "development history consolidated into this commit; the training and inference "
        "code for this checkpoint is unchanged here."
        if git_commit and git_commit != source_commit
        else ""
    )
    return (
        "This fine-tuned checkpoint is distributed under AGPL-3.0-only under the applicable "
        "Ultralytics terms.\n\n"
        "Corresponding source: the training and evaluation source for this checkpoint is "
        f"{SOURCE_REPOSITORY_URL} at commit `{source_commit}`.{trained} The upstream Ultralytics "
        f"source is at {ULTRALYTICS_SOURCE_URL} under the same licence. See notices/ultralytics.md; "
        "the MS COCO provenance of the pretrained initialization is in notices/MS-COCO.md.\n\n"
        f"Training data attribution: external cabbage dataset, DOI {DATASET_DOI}, CC BY 4.0; "
        "see notices/cabbage-dataset.md.\n"
    )


def _license_text(root: Path, model_name: str) -> str:
    candidate = root / ("LICENSE" if model_name in TORCHVISION_MODELS else "LICENSES/AGPL-3.0-only.txt")
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return "Apache License 2.0\n" if model_name in TORCHVISION_MODELS else "GNU AFFERO GENERAL PUBLIC LICENSE\n"


def _write_json(path: Path, payload: object) -> None:
    # LF on every platform: release files are hashed and checked by sha256sum.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _zip_bytes(path: zipfile.ZipFile, member: str, data: bytes) -> None:
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if member.endswith((".pt", ".pth")) else zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    path.writestr(info, data)


def archive_name(run_id: str) -> str:
    """One archive per released run; for a primary run the run id is the model name."""
    return f"cabbage-{run_id}-checkpoints-reproduced-v1.0.0.zip"


def _archive_for_run(
    root: Path, staging: Path, manifest: ReleaseManifest, run_id: str, source_commit: str | None = None
) -> Path:
    records = tuple(record for record in manifest.checkpoints if record.run_id == run_id)
    model_name = records[0].model_name
    archive = staging / archive_name(run_id)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for record in sorted(records, key=lambda item: item.condition):
            source = root / Path(record.source_path)
            member = f"{run_id}/{record.condition}/best.{Path(record.source_path).suffix.lstrip('.') }"
            _zip_bytes(bundle, member, source.read_bytes())
        sidecars = {
            "model_name": model_name,
            "study_scope": records[0].study_scope,
            "checkpoints": [build_sidecar(record, root / "weights/provenance.yaml") for record in sorted(records, key=lambda item: item.condition)],
        }
        _zip_bytes(bundle, f"{run_id}/CHECKPOINTS.json", json.dumps(sidecars, indent=2, sort_keys=True).encode("utf-8"))
        _zip_bytes(bundle, f"{run_id}/MODEL_CARD.md", render_model_card(run_id, records, root).encode("utf-8"))
        _zip_bytes(bundle, f"{run_id}/LICENSE", _license_text(root, model_name).encode("utf-8"))
        _zip_bytes(
            bundle,
            f"{run_id}/THIRD_PARTY_NOTICES.md",
            render_third_party_notices(model_name, root, records[0].git_commit, source_commit).encode("utf-8"),
        )
        for member, relative in sorted(notice_sources(model_name).items()):
            source = root / relative
            if not source.is_file():
                raise ReleaseValidationError(f"required notice file is missing: {relative}")
            _zip_bytes(bundle, f"{run_id}/notices/{member}", source.read_bytes())
    return archive


def build_release(
    root: Path,
    manifest: ReleaseManifest,
    output_dir: Path,
    release_notes: Path | None = None,
    source_commit: str | None = None,
) -> tuple[Path, ...]:
    """Build one deterministic archive per released run, and release metadata, atomically."""
    verify_release_sources(root, manifest)
    root = Path(root)
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        runs = tuple(dict.fromkeys(record.run_id for record in manifest.checkpoints))
        status(f"building release: {len(runs)} run archive(s)")
        archives = tuple(
            _archive_for_run(root, staging, manifest, run_id, source_commit)
            for run_id in progress(runs, description="build release archives", total=len(runs))
        )
        _write_json(staging / "release-manifest.json", {
            "release_tag": manifest.release_tag,
            "dataset_doi": manifest.dataset_doi,
            "checkpoints": [asdict(record) for record in manifest.checkpoints],
        })
        if release_notes is not None:
            # Normalized rather than copied, so a CRLF checkout builds the same bytes.
            (staging / "RELEASE_NOTES.md").write_text(
                Path(release_notes).read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        files_for_sum = sorted(path for path in staging.iterdir() if path.is_file())
        sums = "".join(f"{sha256_file(path)}  {path.name}\n" for path in files_for_sum)
        # A CRLF line makes `sha256sum -c` look for a filename ending in "\r".
        (staging / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        # Directory replacement is restricted by some Windows endpoint
        # policies even when the destination is absent.  The staging tree is
        # complete and audited at this point, so copy it into the generated
        # destination and remove only that temporary tree.
        shutil.copytree(staging, output_dir)
        shutil.rmtree(staging)
        status(f"release build complete: output={output_dir}")
        return tuple(output_dir / path.name for path in (*archives, output_dir / "release-manifest.json", output_dir / "SHA256SUMS"))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def audit_release_directory(output_dir: Path, manifest: ReleaseManifest) -> None:
    """Verify archive names, members, hashes, and release-boundary exclusions."""
    validate_release_manifest(manifest)
    output_dir = Path(output_dir)
    archives = sorted(output_dir.glob("*.zip"))
    runs = tuple(dict.fromkeys(record.run_id for record in manifest.checkpoints))
    if len(archives) != len(runs):
        raise ReleaseValidationError(f"expected {len(runs)} archives, found {len(archives)}")
    for run_id in progress(runs, description="audit release archives", total=len(runs)):
        archive = output_dir / archive_name(run_id)
        if not archive.is_file() or archive.stat().st_size >= 2 * 1024 * 1024 * 1024:
            raise ReleaseValidationError(f"missing or oversized archive: {archive.name}")
        records = tuple(record for record in manifest.checkpoints if record.run_id == run_id)
        model_name = records[0].model_name
        expected_members = {
            *(f"{run_id}/{record.condition}/best.{Path(record.source_path).suffix.lstrip('.') }" for record in records),
            f"{run_id}/CHECKPOINTS.json",
            f"{run_id}/MODEL_CARD.md",
            f"{run_id}/LICENSE",
            f"{run_id}/THIRD_PARTY_NOTICES.md",
            *(f"{run_id}/notices/{member}" for member in notice_sources(model_name)),
        }
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
            forbidden = [name for name in names if "last.pt" in name or name.endswith(("args.yaml", ".csv", ".png", ".jpg", ".pdf")) or Path(name).is_absolute() or ".." in Path(name).parts]
            if forbidden:
                raise ReleaseValidationError(f"forbidden archive member: {forbidden[0]}")
            if names != expected_members:
                raise ReleaseValidationError(f"archive members differ for {run_id}")
            for record in records:
                member = f"{run_id}/{record.condition}/best.{Path(record.source_path).suffix.lstrip('.') }"
                if hashlib.sha256(bundle.read(member)).hexdigest() != record.sha256:
                    raise ReleaseValidationError(f"checkpoint hash mismatch in archive: {member}")
    for required in ("release-manifest.json", "SHA256SUMS"):
        if not (output_dir / required).is_file():
            raise ReleaseValidationError(f"missing release metadata: {required}")
    sums = (output_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in sums:
        digest, name = line.split("  ", 1)
        if digest != sha256_file(output_dir / name):
            raise ReleaseValidationError(f"SHA-256SUMS mismatch: {name}")
