from pathlib import Path

import pytest

from cabbage_detection.publication_boundary import audit_publication_boundary


POLICY = Path("configs/publication-boundary.yaml")


def test_rejects_published_result_tree(tmp_path: Path) -> None:
    findings = audit_publication_boundary(
        tmp_path,
        ["results/published/detection_metrics.csv"],
        POLICY,
    )
    assert any("published result" in item.lower() for item in findings)


def test_rejects_publication_pdf(tmp_path: Path) -> None:
    findings = audit_publication_boundary(tmp_path, ["docs/My_paper.pdf"], POLICY)
    assert any("publication PDF" in item for item in findings)


@pytest.mark.parametrize("prefix, asset", [("paper", "checkpoints"), ("published", "model-weights"), ("publication", "weight")])
def test_rejects_publication_model_asset_text(tmp_path: Path, prefix: str, asset: str) -> None:
    path = tmp_path / "README.md"
    path.write_text(f"{prefix}-{asset}-v1.0.0", encoding="utf-8")
    findings = audit_publication_boundary(tmp_path, ["README.md"], POLICY)
    assert any("publication model asset" in item.lower() for item in findings)


def test_allows_bibliographic_doi_in_reference(tmp_path: Path) -> None:
    path = tmp_path / "docs/publication-reference.md"
    path.parent.mkdir(parents=True)
    path.write_text("DOI 10.1109/" + "KST65016.2025.11003298", encoding="utf-8")
    assert audit_publication_boundary(tmp_path, ["docs/publication-reference.md"], POLICY) == ()


def test_rejects_ieee_doi_outside_approved_reference(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("10.1109/" + "KST65016.2025.11003298", encoding="utf-8")
    findings = audit_publication_boundary(tmp_path, ["notes.md"], POLICY)
    assert any("DOI" in item for item in findings)


def test_rejects_unreadable_tracked_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_bytes(b"\xff\xfe")
    findings = audit_publication_boundary(tmp_path, ["notes.md"], POLICY)
    assert any("cannot read" in item.lower() for item in findings)


@pytest.mark.parametrize("path", [
    "runs/published/run/best.pt",
    "weights/releases/" + "paper-" + "checkpoints-v1.0.0.json",
    "weights/releases/" + "paper_" + "checkpoints-v1.0.0.json",
    "weights/releases/" + "published-model-" + "weights.json",
])
def test_rejects_publication_paths(tmp_path: Path, path: str) -> None:
    findings = audit_publication_boundary(tmp_path, [path], POLICY)
    assert findings


@pytest.mark.parametrize("path", ["original_dataset/.gitkeep", "prepared/.gitkeep"])
def test_allows_empty_directory_placeholders(tmp_path: Path, path: str) -> None:
    assert audit_publication_boundary(tmp_path, [path], POLICY) == ()


def test_current_repository_passes_publication_boundary() -> None:
    assert audit_publication_boundary(Path("."), policy_path=POLICY) == ()


def test_rejects_tracked_notebook(tmp_path: Path) -> None:
    findings = audit_publication_boundary(tmp_path, ["analysis.ipynb"], POLICY)
    assert any("notebook" in item.lower() for item in findings)


@pytest.mark.parametrize(
    "path",
    ["original_dataset/.gitkeep", "prepared/.gitkeep", "results/reproduced/.gitkeep"],
)
def test_allows_public_empty_directory_placeholders(path: str) -> None:
    assert audit_publication_boundary(Path("."), [path], POLICY) == ()


def test_publication_policy_rejects_schema_version(tmp_path: Path) -> None:
    policy = POLICY.read_text(encoding="utf-8")
    path = tmp_path / "policy.yaml"
    path.write_text("schema_version: 1\n" + policy, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown publication-boundary key"):
        audit_publication_boundary(tmp_path, [], path)
