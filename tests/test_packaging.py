import tomllib
from pathlib import Path


def project_metadata() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_core_runtime_dependencies_are_declared() -> None:
    project = project_metadata()
    dependencies = tuple(project["dependencies"])
    assert any(item.startswith("PyYAML") for item in dependencies)
    assert any(item.startswith("numpy") for item in dependencies)
    assert any(item.startswith("matplotlib") for item in dependencies)
    assert any(item.startswith("Pillow") for item in dependencies)


def test_framework_and_development_extras_are_declared() -> None:
    extras = project_metadata()["optional-dependencies"]
    assert {"torchvision", "ultralytics", "training", "dev"} <= set(extras)


def test_training_extra_is_exact_union_of_framework_extras() -> None:
    extras = project_metadata()["optional-dependencies"]

    assert "training" in extras
    assert set(extras["training"]) == set(extras["torchvision"]) | set(extras["ultralytics"])
    assert extras["dev"] == ["pytest>=8,<10"]
    # Complexity counting is optional analysis, not part of any training install.
    assert extras["analysis"] == ["torchinfo>=1.8,<2"]


def test_public_project_metadata_has_repository_links() -> None:
    project = project_metadata()
    assert project["urls"]["Repository"] == "https://github.com/panyapat-wongdee/cabbage-detection-counting"
    assert project["authors"]
