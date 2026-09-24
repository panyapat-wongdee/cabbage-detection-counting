import json
from pathlib import Path

import pytest

from cabbage_detection.release_artifacts import MODEL_NAMES, release_run_ids
from cabbage_detection.reproduce import (
    EVALUATE_OPTIONS,
    RUNS,
    ReproductionStopped,
    _run_status,
    plan_preparation,
    plan_run,
)


def test_every_released_run_is_reproduced_from_its_own_config():
    names = [run.name for run in RUNS]
    assert len(names) == len(set(names)) == 19
    assert set(MODEL_NAMES) <= set(names)
    for run in RUNS:
        assert run.config.is_file(), run.config
        assert any(run.name in release_run_ids(model) for model in MODEL_NAMES)


def test_evaluation_uses_the_options_behind_every_reported_row():
    common = ["--split", "all", "--detection-backend", "repository_ap_101", "--ap-confidence", "0.001"]
    assert EVALUATE_OPTIONS["ultralytics"] == common
    assert EVALUATE_OPTIONS["torchvision"] == common + ["--native-score-threshold", "0.001"]


def _write_run(root: Path, name: str, **metadata) -> Path:
    run_dir = root / "runs/reproduced" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_a_run_resumes_from_where_it_stopped(tmp_path: Path):
    run = RUNS[0]
    assert _run_status(run, tmp_path) == "new"
    assert [step[1] for step in plan_run(run, Path("data"), "new")] == [
        "scripts/train.py", "scripts/evaluate.py", "scripts/promote_run.py"]

    run_dir = _write_run(tmp_path, run.name, status="succeeded", result_status="reproduction_candidate")
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"x")
    assert _run_status(run, tmp_path) == "trained"
    (tmp_path / "results/reproduced" / run.name).mkdir(parents=True)
    (tmp_path / "results/reproduced" / run.name / "summary.json").write_text("{}", encoding="utf-8")
    assert _run_status(run, tmp_path) == "evaluated"
    assert [step[1] for step in plan_run(run, Path("data"), "evaluated")] == ["scripts/promote_run.py"]

    (run_dir / "metadata.json").write_text(json.dumps({"result_status": "reproduced_verified"}), encoding="utf-8")
    assert _run_status(run, tmp_path) == "verified"
    assert plan_run(run, Path("data"), "verified") == []


def test_an_unfinished_run_stops_instead_of_being_overwritten(tmp_path: Path):
    run = RUNS[0]
    _write_run(tmp_path, run.name, status="failed", result_status="reproduction_candidate")
    with pytest.raises(ReproductionStopped, match="move it aside"):
        _run_status(run, tmp_path)


def test_dataset_preparation_runs_only_when_the_yolo_view_is_missing(tmp_path: Path):
    assert plan_preparation(Path("original_dataset"), tmp_path)[0][1] == "scripts/prepare_dataset.py"
    (tmp_path / "prepared/fold1_yolo").mkdir(parents=True)
    (tmp_path / "prepared/fold1_yolo/data.yaml").write_text("", encoding="utf-8")
    assert plan_preparation(Path("original_dataset"), tmp_path) == []


def test_a_version_other_than_the_research_lock_is_reported(tmp_path: Path, monkeypatch):
    from importlib import metadata

    from cabbage_detection.reproduce import version_mismatches

    (tmp_path / "requirements.txt").write_text(
        "--extra-index-url https://example.invalid\n"
        "torch==2.5.1+cu118\ntorchvision==0.20.1+cu118\nultralytics==8.4.9\nalbumentations==1.4.21\n",
        encoding="utf-8",
    )
    installed = {"torch": "2.5.1+cu118", "torchvision": "0.20.1+cu118",
                 "ultralytics": "8.4.80", "albumentations": "1.4.21"}
    monkeypatch.setattr(metadata, "version", lambda name: installed[name])
    assert version_mismatches(tmp_path) == [
        "ultralytics: requirements.txt pins 8.4.9, installed 8.4.80"
    ]
    installed["ultralytics"] = "8.4.9"
    assert version_mismatches(tmp_path) == []
