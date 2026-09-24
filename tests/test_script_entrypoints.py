from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_train_entrypoint_imports_src_without_pythonpath() -> None:
    """Direct script execution from the repository root needs no PYTHONPATH."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "scripts/train.py", "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_bootstrap_places_repository_src_first(monkeypatch) -> None:
    """The entrypoint bootstrap must prefer this checkout over another install."""
    source = str(REPOSITORY_ROOT / "src")
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item != source])

    from scripts import _bootstrap

    _bootstrap.add_src_to_path()

    assert sys.path[0] == source
