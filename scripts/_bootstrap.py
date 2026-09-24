"""Make direct script execution work from the repository root."""

from __future__ import annotations

from pathlib import Path
import sys


def add_src_to_path() -> Path:
    """Put this checkout's ``src`` directory first on ``sys.path``."""
    source = (Path(__file__).resolve().parents[1] / "src").resolve()
    if not source.is_dir():
        raise RuntimeError(f"repository source directory does not exist: {source}")
    source_text = str(source)
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)
    return source
