"""Discovery of the unmodified official Mendeley dataset download."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialDatasetLayout:
    """Resolved paths inside one extracted official dataset bundle."""

    download_root: Path
    bundle_root: Path
    images_root: Path
    annotations_path: Path


def discover_official_dataset(download_root: Path) -> OfficialDatasetLayout:
    """Find the single COCO bundle in an extracted Mendeley download.

    The official archive contains a descriptive directory around
    ``annotation.json`` and its sibling ``images`` directory.  Discovery is
    intentionally strict so a typo or a directory containing multiple
    downloads cannot silently select the wrong source.
    """

    root = Path(download_root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset download root does not exist: {root}")
    annotations = sorted(path for path in root.rglob("annotation.json") if path.is_file())
    candidates = [path for path in annotations if (path.parent / "images").is_dir()]
    if len(candidates) != 1:
        if len(candidates) > 1:
            raise ValueError(
                f"found multiple annotation.json files with sibling images directories under {root}"
            )
        raise ValueError(
            f"expected exactly one annotation.json with a sibling images directory under {root}; "
            f"found {len(candidates)} (images directory required)"
        )
    annotation = candidates[0].resolve()
    return OfficialDatasetLayout(
        download_root=root,
        bundle_root=annotation.parent,
        images_root=(annotation.parent / "images").resolve(),
        annotations_path=annotation,
    )
