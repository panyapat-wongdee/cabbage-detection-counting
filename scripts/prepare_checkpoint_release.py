"""Build and audit the local best-checkpoint GitHub Release candidate."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.release_artifacts import (
    ReleaseValidationError,
    audit_release_directory,
    build_release,
    load_release_manifest,
)
from cabbage_detection.progress import status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-notes", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        help="published commit named as the AGPL corresponding source (default: HEAD, which must be pushed)",
    )
    args = parser.parse_args()
    try:
        source_commit = args.source_commit or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.root, capture_output=True, text=True, check=True
        ).stdout.strip()
        manifest = load_release_manifest(args.manifest)
        assets = build_release(args.root, manifest, args.output_dir, args.release_notes, source_commit)
        audit_release_directory(args.output_dir, manifest)
        status(f"release candidate: {len(assets)} files; audit PASS")
        return 0
    except (OSError, ReleaseValidationError, ValueError, subprocess.CalledProcessError) as error:
        status(f"release preparation failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
