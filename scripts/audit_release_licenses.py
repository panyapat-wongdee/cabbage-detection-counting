"""Audit public-release license boundaries without reading model bytes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from _bootstrap import add_src_to_path
except ModuleNotFoundError:  # imported as ``scripts.audit_release_licenses`` in tests
    from scripts._bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.progress import progress, status


MODEL_NAMES = (
    "faster_rcnn",
    "ssd",
    "ssdlite",
    "retinanet",
    "fcos",
    "yolov8n",
    "yolov8m",
    "yolo11n",
    "yolo11m",
    "rt-detr-l",
    "yolo12n",
    "yolo12m",
    "yolo26n",
    "yolo26m",
)
MODEL_SUFFIXES = (".pt", ".pth", ".onnx", ".safetensors")
REQUIRED_IGNORE_PATTERNS = MODEL_SUFFIXES
# Trees that exist only to stage release assets; nothing in them belongs in Git.
GENERATED_RELEASE_PREFIXES = ("dist/",)
# Parts of a run or evaluation directory that hold release assets or bulk
# regenerable output. The small evidence files beside them are tracked on
# purpose, so the rule names the heavy subtrees rather than the whole tree.
GENERATED_RELEASE_FRAGMENTS = (
    "/checkpoints/",
    "/framework/",
    "/records/",
)


def _tracked_paths(root: Path, tracked_manifest: Path | None) -> list[str]:
    if tracked_manifest is not None:
        return [line.strip() for line in tracked_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [item for item in result.stdout.decode().split("\0") if item]


def _has_model_record(registry: str, name: str) -> bool:
    return f"  {name}:" in registry


def _record_text(registry: str, name: str) -> str:
    """Extract one top-level model mapping without requiring a YAML package."""
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\s*$\n(?P<body>.*?)(?=^  [^ \n][^:]*:\s*$|\Z)",
        registry,
    )
    return match.group("body") if match else ""


def audit_release(root: Path, tracked_manifest: Path | None = None) -> list[str]:
    """Return release-boundary findings; never modify or delete files."""
    root = Path(root)
    findings: list[str] = []

    try:
        tracked = _tracked_paths(root, tracked_manifest)
    except (OSError, subprocess.SubprocessError) as error:
        findings.append(f"cannot read tracked-file list: {error}")
        tracked = []

    for path in progress(tracked, description="audit tracked files", total=len(tracked)):
        normalized = path.replace("\\", "/")
        if normalized.lower().endswith(MODEL_SUFFIXES):
            findings.append(f"tracked model binary: {path}")
        if normalized.lower().endswith(".zip"):
            findings.append(f"tracked release archive: {path}")
        lowered = normalized.lower()
        if lowered.startswith(GENERATED_RELEASE_PREFIXES) or (
            lowered.startswith(("runs/reproduced/", "results/reproduced/"))
            and any(fragment in lowered for fragment in GENERATED_RELEASE_FRAGMENTS)
        ):
            findings.append(f"tracked generated release output: {path}")
        if normalized.lower().endswith(("my_paper.pdf", "dataset_paper.pdf")):
            findings.append(f"tracked publication PDF: {path}")

    ignore_path = root / ".gitignore"
    if not ignore_path.exists():
        findings.append("missing .gitignore")
    else:
        ignore_text = ignore_path.read_text(encoding="utf-8")
        for pattern in REQUIRED_IGNORE_PATTERNS:
            if pattern not in ignore_text:
                findings.append(f".gitignore missing model pattern: {pattern}")

    required_files = {
        "LICENSE": "Apache License",
        "NOTICE": "pretrained weights",
        "LICENSES/AGPL-3.0-only.txt": "GNU AFFERO GENERAL PUBLIC LICENSE",
        "LICENSES/BSD-3-Clause-torchvision.txt": "BSD 3-Clause License",
        "weights/provenance.yaml": "models:",
    }
    contents: dict[str, str] = {}
    for relative, marker in required_files.items():
        path = root / relative
        if not path.exists():
            findings.append(f"missing release file: {relative}")
            continue
        try:
            contents[relative] = path.read_text(encoding="utf-8")
        except OSError as error:
            findings.append(f"cannot read release file {relative}: {error}")
            continue
        if marker.lower() not in contents[relative].lower():
            findings.append(f"release file missing required text: {relative}")

    registry = contents.get("weights/provenance.yaml", "")
    for name in MODEL_NAMES:
        if not _has_model_record(registry, name):
            findings.append(f"registry missing model: {name}")
        else:
            record = _record_text(registry, name)
            expected_license = (
                "Apache-2.0"
                if name in {"faster_rcnn", "ssd", "ssdlite", "retinanet", "fcos"}
                else "AGPL-3.0-only"
            )
            expected_notice = (
                "BSD-3-Clause"
                if name in {"faster_rcnn", "ssd", "ssdlite", "retinanet", "fcos"}
                else "AGPL-3.0-only"
            )
            if f"license: {expected_license}" not in record:
                findings.append(f"registry {name} has wrong fine-tuned license")
            if f"- {expected_notice}" not in record:
                findings.append(f"registry {name} is missing required notice: {expected_notice}")
    if registry and not re.search(
        r"(?ms)^  ssdlite:\s*$.*?^\s+study_scope:\s+supplementary_unreported\s*$",
        registry,
    ):
        findings.append("registry SSDLite study scope is not supplementary_unreported")
    for marker in (
        "source_url:",
        "training_dataset: MS COCO",
        "redistributed: false",
        "distribution: external_release_asset",
    ):
        if marker not in registry:
            findings.append(f"registry missing required field text: {marker}")

    manifest_path = root / "weights/releases/reproduced-checkpoints-v1.0.0.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = manifest.get("checkpoints", [])
            if manifest.get("release_tag") != "reproduced-checkpoints-v1.0.0":
                findings.append("checkpoint manifest has wrong release tag")
            if not records:
                findings.append("checkpoint manifest must contain records")
            if manifest.get("dataset_doi") != "10.17632/5cp2dyjczk.2":
                findings.append("checkpoint manifest has wrong dataset DOI")
            licenses = {record.get("fine_tuned_license") for record in records}
            if not {"Apache-2.0", "AGPL-3.0-only"}.issubset(licenses):
                findings.append("checkpoint manifest is missing one license family")
            if not any(record.get("study_scope") == "supplementary_unreported" for record in records):
                findings.append("checkpoint manifest is missing supplementary SSDLite scope")
        except (OSError, ValueError, TypeError, AttributeError) as error:
            findings.append(f"cannot read checkpoint manifest: {error}")

    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--tracked-manifest", type=Path)
    args = parser.parse_args()
    findings = audit_release(args.root, args.tracked_manifest)
    if findings:
        status(f"release license audit: FAIL ({len(findings)} finding(s))")
        print("\n".join(findings))
        return 1
    status("release license audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
