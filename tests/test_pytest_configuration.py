from pathlib import Path
import tomllib


def test_pytest_configuration_exposes_repository_root_for_repo_scripts() -> None:
    project_root = Path(__file__).parents[1]
    config = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    pythonpath = config["tool"]["pytest"]["ini_options"].get("pythonpath", [])

    assert "." in pythonpath
    assert "src" in pythonpath
