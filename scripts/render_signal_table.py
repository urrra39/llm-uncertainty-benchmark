"""Authoritative signal-coverage table (Part B2).

Every registry entry, its status, and the run it was scored in — generated
from the registry plus the two committed results files, so the counts README
quotes ("27 registered", "22 distinct scored", "5 duplicates") cannot drift
from the code. Embedded in AUDIT_RESPONSE.md between SIGNAL_TABLE markers;
pinned by test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import importlib

    for module in (
        "unc_bench.signals.consistency",
        "unc_bench.signals.logprob_signals",
        "unc_bench.signals.trivial",
        "unc_bench.signals.verification",
    ):
        importlib.import_module(module)
    from unc_bench.signals.base import registry

    names = set(registry())
    assert names, "signal registry is empty; imports above must register every family"
    assert any(name.startswith("c_") for name in names), "family C missing from registry"

    run2 = json.loads((REPO_ROOT / "results_run2_withdrawn.json").read_text())
    run2b = json.loads((REPO_ROOT / "results_run2b.json").read_text())
    scored_run2 = set(run2["views"]["primary"]["signals"])
    scored_run2b = set(run2b["views"]["primary"]["signals"])
    lines = [
        "| signal | family | rank-equivalent to | run #2 | run #2b |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(registry()):
        spec = registry()[name]
        dup = spec.rank_equivalent_to or "—"
        lines.append(
            f"| {name} | {spec.family} | {dup} | "
            f"{'scored' if name in scored_run2 else '—'} | "
            f"{'scored' if name in scored_run2b else '—'} |"
        )
    scored = sorted(scored_run2b)
    lines.append("")
    lines.append(
        f"Tallies: {len(registry())} registered; "
        f"{len([n for n in scored if not registry()[n].rank_equivalent_to])} "
        f"distinct scored in run #2b; "
        f"{len([n for n in scored if registry()[n].rank_equivalent_to])} "
        f"duplicates scored alongside."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    print(render(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
