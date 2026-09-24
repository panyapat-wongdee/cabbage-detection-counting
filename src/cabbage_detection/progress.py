"""Small terminal progress and durable status helpers.

Progress bars are intentionally terminal-only.  Human-readable stage messages
can additionally be persisted in a run's ``console.log`` without capturing or
duplicating framework-owned logs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # lightweight audit jobs may install only their YAML dependency
    _tqdm = None


Item = TypeVar("Item")


def _format_progress_fields(fields: dict[str, object]) -> dict[str, str]:
    """Format live progress fields while keeping the bar compact and readable."""

    formatted: dict[str, str] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            if key == "lr":
                formatted[key] = f"{value:.6g}"
            elif key == "gpu_mem":
                formatted[key] = f"{value:.2f}G"
            else:
                formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = str(value)
    return formatted


class ProgressTask:
    """Small stateful wrapper around an optional tqdm progress bar."""

    def __init__(self, bar: Any | None) -> None:
        self._bar = bar

    def advance(self, **fields: object) -> None:
        """Update the bar with current metrics and advance one unit."""

        if self._bar is None:
            return
        self._bar.set_postfix(_format_progress_fields(fields), refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        """Close the underlying bar, if one was created."""

        if self._bar is not None:
            self._bar.close()


@contextmanager
def progress_task(
    *,
    description: str,
    total: int | None,
    unit: str,
) -> Iterator[ProgressTask]:
    """Yield a stateful terminal progress task with a safe no-tqdm fallback."""

    bar = None
    if _tqdm is not None:
        bar = _tqdm(
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            disable=not sys.stderr.isatty(),
        )
    task = ProgressTask(bar)
    try:
        yield task
    finally:
        task.close()


def status(message: str, *, log_file: Path | None = None) -> None:
    """Write one human-readable status line to stderr and, optionally, a run log."""

    text = str(message)
    if _tqdm is None or not hasattr(_tqdm, "write"):
        print(text, file=sys.stderr, flush=True)
    else:
        _tqdm.write(text, file=sys.stderr)
    sys.stderr.flush()
    if log_file is not None:
        destination = Path(log_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")


def progress(
    items: Iterable[Item],
    *,
    description: str,
    total: int | None = None,
) -> Iterable[Item]:
    """Wrap an iterable in a terminal progress bar.

    ``tqdm`` already closes the bar when iteration completes or is interrupted;
    disabling it for non-interactive stderr keeps batch logs clean while status
    messages remain visible and durable.
    """

    if _tqdm is None:
        return items
    return _tqdm(
        items,
        desc=description,
        total=total,
        unit="item",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
