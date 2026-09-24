"""Build the reproduced-checkpoint release manifest from promoted runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.progress import status
from cabbage_detection.release_artifacts import (
    RELEASE_MODELS,
    ReleaseValidationError,
    build_release_manifest,
    release_run_ids,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs") / "reproduced",
        help="directory holding one promoted run per model (default: runs/reproduced)",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        type=Path,
        help="a single run directory; repeatable (default: every promoted run found)",
    )
    parser.add_argument(
        "--condition",
        default="augmented",
        choices=("augmented", "no_augmentation"),
        help="training condition these runs represent (default: augmented)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        runs = args.runs or [
            path
            for path in sorted(args.runs_root.iterdir())
            # Every model's primary run and its named variant runs
            # (<model>-detector512); anything else, such as smoke, is skipped.
            if path.is_dir() and (path / "metadata.json").is_file()
            and any(path.name in release_run_ids(model) for model in RELEASE_MODELS)
        ]
        payload = build_release_manifest(args.root, runs, args.condition)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        status(f"release manifest written: {args.output} ({len(payload['checkpoints'])} checkpoints)")
        return 0
    except (OSError, ReleaseValidationError, ValueError, KeyError) as error:
        status(f"manifest build failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
