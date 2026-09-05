"""Render the README status header from `results.json` (Part F3).

A reader must not have to scroll to learn which run is primary, its n, its
label-quality status, and whether the gates passed. This script derives that
block from the committed results file plus the committed validation CSV, so
the header cannot drift from the artifacts. Checked by
`test_readme_header_matches_generated` (regeneration must be a no-op):

    uv run python scripts/render_readme_header.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    payload = json.loads((REPO_ROOT / "results.json").read_text(encoding="utf-8"))
    view = payload["views"]["primary"]
    gates = payload["validity_gates"]
    gate_state = (
        "ALL THREE VALIDITY GATES PASS"
        if gates.get("all_passed")
        else f"VALIDITY FAILED: {', '.join(gates.get('failed', []))}"
    )
    csv_path = REPO_ROOT / "data" / "human_validation_sample.csv"
    labelled = 0
    total = 0
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                total += 1
                if (row.get("human_label") or "").strip():
                    labelled += 1
    random_entry = view["signals"]["t_random"]["auroc"]
    return "\n".join(
        [
            f"Primary run: {payload['run_name']} (n={view['n']}, "
            f"{view['n_incorrect']} incorrect / {view['n_correct']} correct).",
            "",
            f"{gate_state} as recorded in this file "
            f"(t_random {random_entry['point']:.3f} "
            f"[{random_entry['ci_low']:.3f}, {random_entry['ci_high']:.3f}]).",
            f"Label quality: {labelled}/{total} human-labelled — "
            "the correctness of the label set is unmeasured.",
        ]
    )


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
