"""Fail-closed checks for the public publication-content boundary."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from collections.abc import Iterable
from typing import Any

import yaml

from .progress import progress


TEXT_SUFFIXES = {
    ".c",
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _normalise(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _load_policy(policy_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(policy_path).read_text(encoding="utf-8"))
    required_keys = {
        "allowed_doi_paths",
        "allowed_dataset_doi_prefixes",
        "forbidden_path_prefixes",
        "forbidden_path_fragments",
        "publication_asset_qualifiers",
        "model_asset_terms",
        "forbidden_suffixes",
        "forbidden_text_markers",
        "publication_doi",
        "dataset_doi",
    }
    if not isinstance(raw, dict):
        raise ValueError("publication-boundary policy must be a mapping")
    missing = sorted(required_keys - set(raw))
    if missing:
        raise ValueError(f"publication-boundary policy missing key(s): {', '.join(missing)}")
    unknown = sorted(set(raw) - required_keys)
    if unknown:
        raise ValueError(f"unknown publication-boundary key(s): {', '.join(unknown)}")
    for key in (
        "allowed_doi_paths",
        "allowed_dataset_doi_prefixes",
        "forbidden_path_prefixes",
        "forbidden_path_fragments",
        "publication_asset_qualifiers",
        "model_asset_terms",
        "forbidden_suffixes",
        "forbidden_text_markers",
    ):
        if not isinstance(raw.get(key), list) or not all(isinstance(item, str) for item in raw[key]):
            raise ValueError(f"publication-boundary policy field {key} must be a string list")
    for key in ("publication_asset_qualifiers", "model_asset_terms"):
        if not raw[key] or any(not item.strip() for item in raw[key]):
            raise ValueError(f"publication-boundary policy field {key} must contain nonempty terms")
    if not isinstance(raw.get("publication_doi"), str) or not isinstance(raw.get("dataset_doi"), str):
        raise ValueError("publication-boundary policy DOI values must be strings")
    return raw


def _tracked_paths(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return tuple(item for item in result.stdout.decode("utf-8").split("\0") if item)


def audit_publication_boundary(
    root: Path,
    tracked_paths: Iterable[str] | None = None,
    policy_path: Path = Path("configs/publication-boundary.yaml"),
) -> tuple[str, ...]:
    """Return sorted findings for tracked files that cross the public boundary."""

    root = Path(root)
    policy_candidate = Path(policy_path)
    if not policy_candidate.is_absolute():
        policy_candidate = (
            root / policy_candidate
            if (root / policy_candidate).is_file()
            else policy_candidate
        )
    policy = _load_policy(policy_candidate)
    paths = tuple(tracked_paths) if tracked_paths is not None else _tracked_paths(root)
    allowed_doi_paths = {_normalise(path) for path in policy["allowed_doi_paths"]}
    # Generated run evidence must record dataset provenance, so the dataset DOI
    # is allowed under these prefixes. The publication DOI is not: it stays
    # restricted to the exact curated citation paths above.
    allowed_dataset_doi_prefixes = tuple(
        _normalise(prefix) for prefix in policy["allowed_dataset_doi_prefixes"]
    )
    policy_relative = _normalise(str(Path(policy_path).resolve().relative_to(root.resolve()))) if Path(policy_path).resolve().is_relative_to(root.resolve()) else ""
    findings: set[str] = set()
    publication_doi = re.escape(policy["publication_doi"])
    dataset_doi = re.escape(policy["dataset_doi"])
    qualifiers = "|".join(re.escape(term) for term in policy["publication_asset_qualifiers"])
    assets = "|".join(re.escape(term) for term in policy["model_asset_terms"])
    publication_asset_pattern = re.compile(
        rf"(?<![a-z0-9])(?:{qualifiers})[-_ /]+(?:model[-_ /]+)?(?:{assets})(?![a-z0-9])",
        re.IGNORECASE,
    )

    for raw_path in progress(paths, description="audit publication boundary", total=len(paths)):
        path = _normalise(raw_path)
        lowered = path.lower()
        for prefix in policy["forbidden_path_prefixes"]:
            if lowered.startswith(_normalise(prefix).lower()) and not lowered.endswith("/.gitkeep"):
                label = "published result" if "published" in lowered else "external/generated artifact"
                findings.add(f"forbidden {label} path: {raw_path}")
        for fragment in policy["forbidden_path_fragments"]:
            if fragment.lower() in lowered:
                findings.add(f"forbidden path fragment: {raw_path}")
        if publication_asset_pattern.search(path):
            findings.add(f"forbidden publication model asset path: {raw_path}")
        if any(lowered.endswith(suffix.lower()) for suffix in policy["forbidden_suffixes"]):
            if lowered.endswith(".ipynb"):
                findings.add(f"tracked notebook artifact: {raw_path}")
            else:
                findings.add(f"tracked publication PDF: {raw_path}")

        file_path = root / Path(path)
        if not file_path.is_file() or file_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path == policy_relative:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            findings.add(f"cannot read tracked text {raw_path}: {error}")
            continue
        for marker in policy["forbidden_text_markers"]:
            if marker.lower() in text.lower():
                findings.add(f"forbidden publication marker in {raw_path}: {marker}")
        if publication_asset_pattern.search(text):
            findings.add(f"forbidden publication model asset text in {raw_path}")
        if re.search(publication_doi, text) and path not in allowed_doi_paths:
            findings.add(f"IEEE publication DOI outside approved citation path: {raw_path}")
        if (
            re.search(dataset_doi, text)
            and path not in allowed_doi_paths
            and not (allowed_dataset_doi_prefixes and path.startswith(allowed_dataset_doi_prefixes))
        ):
            findings.add(f"dataset DOI outside approved attribution path: {raw_path}")

    return tuple(sorted(findings))
