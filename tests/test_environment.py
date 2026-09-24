from cabbage_detection.environment import collect_environment


def test_collect_environment_records_core_runtime():
    info = collect_environment()
    payload = info.to_dict()

    assert payload["python"]
    assert payload["platform"]
    assert "cuda_available" in payload


def test_collect_environment_uses_null_for_missing_optional_package(monkeypatch):
    monkeypatch.setattr(
        "cabbage_detection.environment._package_version",
        lambda name: None if name == "missing-package" else "1.0",
    )

    info = collect_environment(package_names=("missing-package",))

    assert info.packages["missing-package"] is None
