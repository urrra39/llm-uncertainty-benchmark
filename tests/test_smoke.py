"""Scaffold-level check: the package imports and exposes a version."""

from pathlib import Path

import unc_bench


def test_version_is_a_string() -> None:
    assert isinstance(unc_bench.__version__, str)
    assert unc_bench.__version__.count(".") == 2


def test_generate_lock_roundtrip_and_refusal(tmp_path: Path) -> None:
    """D36: a second concurrent generate refuses; a stale lock is taken over."""
    import os

    from unc_bench.stages.generate import (
        acquire_generate_lock,
        release_generate_lock,
    )

    root = tmp_path / "artifacts"
    first = acquire_generate_lock(root)
    assert first.exists()
    try:
        acquire_generate_lock(root)
    except RuntimeError as exc:
        assert "refusing" in str(exc)
    else:
        raise AssertionError("second concurrent acquire must refuse")
    release_generate_lock(first)
    assert not first.exists()
    # Stale lock (dead pid) is taken over, corrupt content too.
    stale = root / ".generate.lock"
    stale.write_text("999999999", encoding="utf-8")
    second = acquire_generate_lock(root)
    assert second.read_text(encoding="utf-8").strip() == str(os.getpid())
    release_generate_lock(second)
    stale.write_text("not-a-pid", encoding="utf-8")
    third = acquire_generate_lock(root)
    release_generate_lock(third)
    assert not third.exists()
