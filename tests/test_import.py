def test_package_exposes_version():
    import cabbage_detection

    assert isinstance(cabbage_detection.__version__, str)
