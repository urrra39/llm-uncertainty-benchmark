"""Render the null-vs-significance tension paragraph from results (Part A1).

The lead states a null result while the paired-bootstrap test can still
reject some signals. Both are defensible and the document must say why in one
place: the paired bootstrap on differences exploits between-signal
correlation and has more power than comparing marginal intervals. The band
count and the rejected names are read from the primary results file
(`results_run2b_fixedlabels.json`, the published run), not hand-counted.
Embedded in README between TENSION markers and pinned by test:

    uv run python scripts/render_tension_paragraph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def render(results_path: str = "results_run2b_fixedlabels.json") -> str:
    payload: dict[str, Any] = json.loads((REPO_ROOT / results_path).read_text())
    view = payload["views"]["primary"]
    stratified = view["stratified"]["signals"]
    total = len(stratified)
    leader = max(stratified, key=lambda n: stratified[n]["point"])
    low, high = stratified[leader]["ci_low"], stratified[leader]["ci_high"]
    band = sum(1 for e in stratified.values() if e["ci_low"] <= high and e["ci_high"] >= low)
    rejected = sorted(
        c["name"] for c in view["significance"]["comparisons"] if c.get("significant_holm_distinct")
    )
    if rejected:
        tail = (
            f"Here: {band} of {total} signals' intervals overlap the leader's band "
            f"[{low:.3f}, {high:.3f}], while "
            + (" and ".join(f"`{n}`" for n in rejected))
            + (
                " is significantly worse after Holm"
                if len(rejected) == 1
                else " are significantly worse after Holm"
            )
            + "."
        )
    else:
        tail = (
            f"Here: {band} of {total} signals' intervals overlap the leader's band "
            f"[{low:.3f}, {high:.3f}], and no signal is significantly worse after Holm."
        )
    return "\n".join(
        [
            "Why the null leads while the significance test can still reject: "
            "the two tests answer different questions. Marginal-interval overlap "
            "asks whether two point estimates can be told apart on their own — "
            "the conservative read, and why the null leads. The paired bootstrap "
            "on AUROC differences asks whether one signal beats another on "
            "the same rows, exploiting their correlation; that is the test "
            "with the power to reject. " + tail,
            "",
        ]
    )


def main() -> int:
    print(render(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
