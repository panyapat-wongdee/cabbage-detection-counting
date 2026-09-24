from __future__ import annotations

from pathlib import Path

import cabbage_detection.progress as progress_module
from cabbage_detection.progress import progress, progress_task, status


def test_status_writes_terminal_and_log_message(capsys, tmp_path: Path):
    log_path = tmp_path / "logs" / "console.log"

    status("stage started", log_file=log_path)

    assert "stage started" in capsys.readouterr().err
    assert log_path.read_text(encoding="utf-8") == "stage started\n"


def test_progress_preserves_items_and_accepts_total():
    assert list(progress([1, 2, 3], description="items", total=3)) == [1, 2, 3]


def test_status_and_progress_fallback_without_tqdm(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setattr(progress_module, "_tqdm", None)
    log_path = tmp_path / "console.log"

    status("fallback status", log_file=log_path)

    assert list(progress((1, 2), description="items", total=2)) == [1, 2]
    assert "fallback status" in capsys.readouterr().err
    assert log_path.read_text(encoding="utf-8") == "fallback status\n"


def test_progress_task_updates_live_fields_and_closes(monkeypatch):
    events: list[tuple[str, object]] = []

    class FakeBar:
        def set_postfix(self, fields, *, refresh):
            events.append(("postfix", (fields, refresh)))

        def update(self, amount):
            events.append(("update", amount))

        def close(self):
            events.append(("close", None))

    def fake_tqdm(*, total, desc, unit, dynamic_ncols, disable):
        events.append(("create", (total, desc, unit, dynamic_ncols, disable)))
        return FakeBar()

    monkeypatch.setattr(progress_module, "_tqdm", fake_tqdm)
    monkeypatch.setattr(progress_module.sys.stderr, "isatty", lambda: True)

    with progress_task(description="train 1/2", total=2, unit="batch") as task:
        task.advance(loss=1.23456, avg_loss=2.0, lr=0.001, images=8, skipped=0, gpu_mem=None)

    assert events[0] == ("create", (2, "train 1/2", "batch", True, False))
    assert events[1] == (
        "postfix",
        ({"loss": "1.2346", "avg_loss": "2.0000", "lr": "0.001", "images": "8", "skipped": "0"}, False),
    )
    assert events[2] == ("update", 1)
    assert events[-1] == ("close", None)


def test_progress_task_falls_back_without_tqdm(monkeypatch):
    monkeypatch.setattr(progress_module, "_tqdm", None)

    with progress_task(description="train", total=1, unit="batch") as task:
        task.advance(loss=1.0)

    task.close()
