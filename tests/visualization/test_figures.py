import json
import subprocess
import sys
from pathlib import Path

import pytest

from cabbage_detection.visualization.figures import plot_model_comparison

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(model, framework, scope, map50_95, f1):
    return {
        "model": model,
        "source": model,
        "framework": framework,
        "study_scope": scope,
        "detection": {"map50_95": map50_95},
        "counting": {"f1": f1},
    }


def test_model_comparison_renders_every_model(tmp_path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "split": "test",
                "image_count": 2,
                "models": [
                    _record("fcos", "torchvision", "paper_model", 0.7, 0.95),
                    _record("yolo11n", "ultralytics", "paper_model", 0.69, 0.95),
                    _record("ssdlite", "torchvision", "supplementary_unreported", 0.47, 0.69),
                ],
            }
        ),
        encoding="utf-8",
    )

    complexity = tmp_path / "complexity.json"
    complexity.write_text(
        json.dumps(
            {
                "models": [
                    {"model": name, "run": name, "params": params, "gflops": gflops,
                     "run_gflops": gflops, "flops_input": [512, 512]}
                    for name, params, gflops in (
                        ("fcos", 32_000_000, 102.7),
                        ("yolo11n", 2_600_000, 4.1),
                        ("ssdlite", 2_200_000, 2.1),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )

    output = plot_model_comparison(comparison, complexity, tmp_path / "figure.png")

    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.with_suffix(".svg").is_file()
    assert not output.with_suffix(".pdf").exists()


def test_model_comparison_refuses_a_model_without_a_complexity_record(tmp_path):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps({"split": "test", "image_count": 1,
                    "models": [_record("fcos", "torchvision", "paper_model", 0.7, 0.95)]}),
        encoding="utf-8",
    )
    complexity = tmp_path / "complexity.json"
    complexity.write_text(json.dumps({"models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no complexity record"):
        plot_model_comparison(comparison, complexity, tmp_path / "figure.png")


def test_make_figures_entrypoint_runs_without_pythonpath():
    completed = subprocess.run(
        [sys.executable, "scripts/make_figures.py", "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "comparison" in completed.stdout and "detection" in completed.stdout
