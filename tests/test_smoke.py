def test_imports():
    import importlib

    # Basic smoke test to ensure project packages are importable
    for module in ["src", "scripts"]:
        importlib.import_module(module)
