"""Render docs/LABEL_GATE_MAP.md from code constants (Part A1).

One row per human-label gate naming the exact file it reads, the coverage
denominator, the threshold and what labelling unlocks. Generated from
`GATE_SOURCES` in analysis/validity.py plus the run #2b config's resolved
paths — never by hand — so the document cannot drift from the code. Checked
by test_gate_map_matches_generated (regeneration must be a no-op):

    uv run python scripts/render_gate_map.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def render(config_name: str = "configs/run2b_clean.yaml") -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from unc_bench.analysis.validity import GATE_SOURCES
    from unc_bench.config import Config

    cfg = Config.load(REPO_ROOT / config_name)
    resolved = {
        "labeling_protocol_validated": str(cfg.paths.human_validation_csv),
        "human_label_coverage": (
            str(cfg.paths.fuzzy_decided_csv)
            if cfg.paths.fuzzy_decided_csv is not None
            else "(unconfigured)"
        ),
    }
    lines = [
        "# Label gate map",
        "",
        f"Generated from `GATE_SOURCES` plus `{config_name}` — edit the code, "
        "not this file. Pinned by test.",
        "",
        "| gate | reads | denominator | threshold | labelling it unlocks |",
        "|---|---|---|---|---|",
    ]
    for gate, meta in GATE_SOURCES.items():
        lines.append(
            f"| {gate} | {resolved[gate]} ({meta['reads']}) "
            f"| {meta['denominator']} | {meta['threshold']} | {meta['unlocks']} |"
        )
    lines.append("")
    lines.append(
        "Every file above belongs to the run being gated. Labelling withdrawn "
        "rows earns no gate credit by construction."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    print(render(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
