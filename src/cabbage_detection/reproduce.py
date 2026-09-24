"""Train, evaluate, and promote every released run, then build the comparison artifacts.

One command runs the whole study from a clean checkout:

    python scripts/reproduce_all.py --dataset original_dataset

Each run goes through the same three repository commands a person would run by
hand (``train.py``, ``evaluate.py``, ``promote_run.py``) with the evaluation
options every reported result uses, so the orchestration adds no behaviour of
its own. It resumes: a promoted run is skipped, and a run interrupted after
training continues at evaluation or promotion. A run directory it cannot
account for stops the sequence instead of being overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

RUNS_ROOT = Path("runs/reproduced")
REQUIREMENTS = Path("requirements.txt")
# The packages whose version changes model training or scoring; every run
# records them, and a reproduction must use the versions requirements.txt pins.
PINNED_PACKAGES = ("torch", "torchvision", "ultralytics", "albumentations")
RESULTS_ROOT = Path("results/reproduced")
PREPARED_YOLO = Path("prepared/fold1_yolo")
SPLIT_MANIFEST = Path("splits/fold1_recovered.csv")

# The evaluation options behind every reported row. Torchvision detectors
# also lower their own score gate so AP covers the full score range; that is
# recorded as a deviation in each split's metadata.
EVALUATE_OPTIONS = {
    "torchvision": ["--split", "all", "--detection-backend", "repository_ap_101",
                    "--ap-confidence", "0.001", "--native-score-threshold", "0.001"],
    "ultralytics": ["--split", "all", "--detection-backend", "repository_ap_101",
                    "--ap-confidence", "0.001"],
}


@dataclass(frozen=True)
class Run:
    name: str
    config: Path
    framework: str


RUNS: tuple[Run, ...] = (
    *(Run(name, Path(f"configs/torchvision/{name}.yaml"), "torchvision")
      for name in ("faster_rcnn", "ssd", "retinanet", "fcos", "ssdlite")),
    *(Run(name, Path(f"configs/ultralytics/{name}.yaml"), "ultralytics")
      for name in ("yolov8n", "yolov8m", "yolo11n", "yolo11m", "rt-detr-l",
                   "yolo12n", "yolo12m", "yolo26n", "yolo26m")),
    *(Run(f"{name}-detector512", Path(f"configs/torchvision/detector_input_512/{name}.yaml"), "torchvision")
      for name in ("faster_rcnn", "ssd", "retinanet", "fcos", "ssdlite")),
)


class ReproductionStopped(RuntimeError):
    """Raised when a run is in a state the orchestration must not guess about."""


def _python(script: str, *arguments: str) -> list[str]:
    return [sys.executable, f"scripts/{script}", *arguments]


def _run_status(run: Run, root: Path) -> str:
    """Classify a run as new, trained, evaluated, or verified."""
    run_dir = root / RUNS_ROOT / run.name
    results_dir = root / RESULTS_ROOT / run.name
    if not run_dir.exists():
        return "new"
    metadata_path = run_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproductionStopped(
            f"{run_dir} exists without readable metadata; move it aside to retrain {run.name}"
        ) from error
    if metadata.get("result_status") == "reproduced_verified":
        return "verified"
    if metadata.get("status") != "succeeded" or not (run_dir / "checkpoints" / "best.pt").is_file():
        raise ReproductionStopped(
            f"{run_dir} did not finish training (status {metadata.get('status')!r}); "
            f"move it aside to retrain {run.name}"
        )
    if not results_dir.exists():
        return "trained"
    if (results_dir / "summary.json").is_file():
        return "evaluated"
    raise ReproductionStopped(
        f"{results_dir} is incomplete; move it aside to re-evaluate {run.name}"
    )


def plan_run(run: Run, dataset: Path, status: str) -> list[list[str]]:
    """The commands that take one run from ``status`` to ``reproduced_verified``."""
    train = _python("train.py", "--config", run.config.as_posix())
    evaluate = _python("evaluate.py", "--run", (RUNS_ROOT / run.name).as_posix(),
                       "--dataset", dataset.as_posix(), *EVALUATE_OPTIONS[run.framework])
    promote = _python("promote_run.py", "--run", (RUNS_ROOT / run.name).as_posix(),
                      "--dataset", dataset.as_posix())
    steps = {"new": [train, evaluate, promote], "trained": [evaluate, promote],
             "evaluated": [promote], "verified": []}
    return steps[status]


def plan_preparation(dataset: Path, root: Path) -> list[list[str]]:
    """Build the ignored Ultralytics dataset view when it is missing."""
    if (root / PREPARED_YOLO / "data.yaml").is_file():
        return []
    return [_python("prepare_dataset.py", "--format", "all", "--dataset-root", dataset.as_posix(),
                    "--manifest", SPLIT_MANIFEST.as_posix(), "--output", PREPARED_YOLO.as_posix(),
                    "--link-mode", "hardlink")]


def plan_reports(runs: Sequence[Run]) -> list[list[str]]:
    """Cross-model tables and complexity counts once every run is verified."""
    measure = ["measure_complexity.py"]
    for run in runs:
        measure += ["--run", (RUNS_ROOT / run.name).as_posix()]
    return [
        _python("compare_models.py", "--split", "test"),
        _python("compare_models.py", "--split", "test", "--variant", "detector512"),
        _python(*measure),
    ]


def version_mismatches(root: Path) -> list[str]:
    """Installed versions that differ from the research lock in requirements.txt."""
    from importlib import metadata

    pinned = {}
    for line in (root / REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        name, separator, version = line.strip().partition("==")
        if separator and name.lower() in PINNED_PACKAGES:
            pinned[name.lower()] = version.strip()
    mismatches = []
    for name in PINNED_PACKAGES:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != pinned.get(name):
            mismatches.append(f"{name}: requirements.txt pins {pinned.get(name)}, installed {installed}")
    return mismatches


def _tracked_changes(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root, capture_output=True, text=True, check=False,
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", type=Path, required=True, help="extracted official dataset root")
    parser.add_argument("--only", action="append", choices=[run.name for run in RUNS],
                        help="restrict to this run; repeatable (default: all nineteen)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--skip-reports", action="store_true",
                        help="stop after the runs; do not rebuild comparison tables or complexity")
    args = parser.parse_args(argv)
    root = Path.cwd()

    mismatches = version_mismatches(root)
    if mismatches:
        message = ("installed packages differ from requirements.txt; install the research lock "
                   "(python -m pip install -r requirements.txt) before training:\n  "
                   + "\n  ".join(mismatches))
        if not args.dry_run:
            parser.error(message)
        print("warning: " + message, file=sys.stderr)

    if not args.dry_run:
        if not (root / args.dataset).exists() and not args.dataset.exists():
            parser.error(f"dataset root not found: {args.dataset}")
        changes = _tracked_changes(root)
        if changes:
            parser.error("tracked files have uncommitted changes; every run records the commit it "
                         "trained at, so commit or discard them first:\n" + changes)

    selected = [run for run in RUNS if not args.only or run.name in args.only]
    commands = plan_preparation(args.dataset, root)
    try:
        for run in selected:
            status = "new" if args.dry_run and not (root / RUNS_ROOT / run.name).exists() else _run_status(run, root)
            print(f"[{run.name}] {status}", flush=True)
            commands += plan_run(run, args.dataset, status)
    except ReproductionStopped as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 1
    if not args.skip_reports and not args.only:
        commands += plan_reports(selected)

    for command in commands:
        print("$ " + " ".join(command[1:] if command[0] == sys.executable else command), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            print(f"stopped: command failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
    return 0
