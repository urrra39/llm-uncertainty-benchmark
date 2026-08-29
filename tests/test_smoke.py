"""Scaffold-level check: the package imports and exposes a version."""

import unc_bench


def test_version_is_a_string() -> None:
    assert isinstance(unc_bench.__version__, str)
    assert unc_bench.__version__.count(".") == 2
