"""Packaging smoke tests: the package installs and imports. Not a conformance suite."""

import importlib.metadata

import cotorch


def test_version_matches_installed_metadata():
    assert cotorch.__version__ == importlib.metadata.version("cotorch")


def test_public_api_is_only_what_is_declared():
    assert cotorch.__all__ == ["__version__", "matmul"]


def test_numpy_is_not_a_runtime_dependency():
    requires = importlib.metadata.requires("cotorch") or []
    runtime = [r for r in requires if "extra ==" not in r]
    assert runtime == []
