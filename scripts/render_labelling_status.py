"""Render the README labelling block from code, not by hand (Part A3).

The stale paragraph advertised the 100-row sample while the gates read the
fuzzy file. This prints what `label-plan` actually outputs — file order, row
counts, both gates — plus the gate names from GATE_SOURCES, so the block
cannot go stale again. Embedded in README between LABELLING markers and
pinned by test:

    uv run python scripts/render_labelling_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from unc_bench.analysis.validity import GATE_SOURCES
    from unc_bench.stages.label_human import DEFAULT_TARGET, RUN_CSVS, label_plan

    plan: dict[str, Any] = label_plan(RUN_CSVS["run2b"], DEFAULT_TARGET)
    rows_to_label = int(plan["rows_to_label"])
    fuzzy_pair = plan["fuzzy_coverage_after"]
    sample_pair = plan["sample_coverage_after"]
    assert isinstance(fuzzy_pair, list) and isinstance(sample_pair, list)
    fuzzy_done, fuzzy_total = int(fuzzy_pair[0]), int(fuzzy_pair[1])
    sample_done, sample_total = int(sample_pair[0]), int(sample_pair[1])
    gates = sorted(GATE_SOURCES)
    return "\n".join(
        [
            f"To open the gates, label {rows_to_label} rows, "
            f"shared-first: `{DEFAULT_TARGET.as_posix()}` "
            f"({fuzzy_done} of {fuzzy_total} "
            f"for `human_label_coverage`), then "
            f"`{RUN_CSVS['run2b'].as_posix()}` "
            f"to {sample_done} of {sample_total} "
            f"for `labeling_protocol_validated`. "
            f"Wall clock: {plan['wall_clock']}. "
            f"Gate names: {', '.join(f'`{g}`' for g in gates)}.",
            "",
        ]
    )


def main() -> int:
    print(render(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
