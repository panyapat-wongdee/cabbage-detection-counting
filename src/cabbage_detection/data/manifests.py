"""Portable CSV split manifests with deterministic integrity digests."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .layout import DatasetLayout


@dataclass(frozen=True)
class SplitManifest:
    rows: tuple[tuple[str, str], ...]
    provenance: str
    digest: str


def _digest(rows: tuple[tuple[str, str], ...]) -> str:
    payload = "\n".join(f"{split},{image_id}" for split, image_id in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(layout: DatasetLayout, provenance: str) -> SplitManifest:
    rows = tuple(
        (split, image_id)
        for split in layout.splits
        for image_id in layout.image_ids(split)
    )
    return SplitManifest(rows=rows, provenance=provenance, digest=_digest(rows))


def write_manifest(manifest: SplitManifest, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image_id", "provenance", "digest"])
        for split, image_id in manifest.rows:
            writer.writerow([split, image_id, manifest.provenance, manifest.digest])


def read_manifest(path: Path) -> SplitManifest:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest must contain at least one row")
    required = {"split", "image_id", "provenance", "digest"}
    if set(rows[0]) != required:
        raise ValueError("manifest columns must be split,image_id,provenance,digest")
    pairs = tuple((row["split"], row["image_id"]) for row in rows)
    provenance = rows[0]["provenance"]
    digest = rows[0]["digest"]
    if any(row["provenance"] != provenance or row["digest"] != digest for row in rows):
        raise ValueError("manifest metadata is inconsistent")
    if _digest(pairs) != digest:
        raise ValueError("manifest digest does not match rows")
    return SplitManifest(rows=pairs, provenance=provenance, digest=digest)
