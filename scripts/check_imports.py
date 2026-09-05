"""Import every module in `src/` on the dev dependency set alone.

CI installs neither torch nor transformers, and every heavy import in this
repo is deferred into constructors and stage functions for exactly that
reason. This script walks `src/unc_bench` and imports each module so an
undeclared top-level heavy import fails the build instead of passing locally
and breaking a fresh clone. Run in CI after mypy:

    uv run python scripts/check_imports.py
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import unc_bench

    failures: list[str] = []
    checked = 0
    for info in pkgutil.walk_packages(unc_bench.__path__, prefix="unc_bench."):
        # `unc_bench.cli` builds the parser at import; importing it is the
        # point, and it must not execute anything.
        try:
            importlib.import_module(info.name)
            checked += 1
        except Exception as exc:  # reporting the failure is the point
            failures.append(f"{info.name}: {exc!r}")
    if failures:
        print(f"{len(failures)} module(s) failed to import:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"imported {checked} modules with dev dependencies only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
