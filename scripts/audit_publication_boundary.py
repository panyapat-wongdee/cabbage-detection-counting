"""Audit tracked files against the citation-only publication boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from cabbage_detection.publication_boundary import audit_publication_boundary
from cabbage_detection.progress import status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--policy", type=Path, default=Path("configs/publication-boundary.yaml"))
    parser.add_argument("--tracked-manifest", type=Path)
    args = parser.parse_args()
    tracked = None
    if args.tracked_manifest is not None:
        tracked = tuple(
            line.strip()
            for line in args.tracked_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    findings = audit_publication_boundary(args.root, tracked, args.policy)
    if findings:
        status(f"publication boundary audit: FAIL ({len(findings)} finding(s))")
        print("\n".join(findings))
        return 1
    status("publication boundary audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
