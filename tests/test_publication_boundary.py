from pathlib import Path

import yaml


def test_publication_reference_is_citation_only() -> None:
    text = Path("docs/publication-reference.md").read_text(encoding="utf-8")
    assert "10.1109/KST65016.2025.11003298" in text
    assert "A Comparative Study of Deep Learning Models" in text
    for forbidden in (
        "![",
        "<img",
        "mAP@50",
        "counting F1",
        "results/published",
        "paper-" + "checkpoints",
    ):
        assert forbidden.lower() not in text.lower()


def test_publication_boundary_states_verified_status_and_manual_review() -> None:
    text = Path("docs/publication-boundary.md").read_text(encoding="utf-8").lower()
    for required in (
        "citation-only",
        "reproduced_verified",
        "manual",
        "supplementary_unreported",
    ):
        assert required in text


def test_dataset_doi_prefix_allowance_does_not_relax_the_publication_doi(tmp_path):
    """Run evidence may carry dataset attribution; it may not carry the IEEE DOI."""
    import subprocess

    from cabbage_detection.publication_boundary import audit_publication_boundary

    policy = yaml.safe_load(Path("configs/publication-boundary.yaml").read_text(encoding="utf-8"))
    assert "runs/reproduced/" in policy["allowed_dataset_doi_prefixes"]

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "publication-boundary.yaml").write_text(
        yaml.safe_dump(policy), encoding="utf-8"
    )
    evidence = tmp_path / "runs" / "reproduced" / "demo"
    evidence.mkdir(parents=True)
    (evidence / "dataset.json").write_text(
        f'{{"doi": "{policy["dataset_doi"]}"}}', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)

    allowed = audit_publication_boundary(
        tmp_path, policy_path=tmp_path / "configs" / "publication-boundary.yaml"
    )
    assert not [item for item in allowed if "dataset DOI" in item]

    (evidence / "dataset.json").write_text(
        f'{{"doi": "{policy["publication_doi"]}"}}', encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    refused = audit_publication_boundary(
        tmp_path, policy_path=tmp_path / "configs" / "publication-boundary.yaml"
    )
    assert any("IEEE publication DOI" in item for item in refused)
