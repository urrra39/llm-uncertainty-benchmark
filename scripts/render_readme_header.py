"""Render the README status header from the primary results file (Part F3).

A reader must not have to scroll to learn which run is primary, its n, its
label-quality status, and whether the gates passed. This script derives that
block from the primary results file plus the committed validation CSV, so
the header cannot drift from the artifacts. Checked by
`test_readme_header_matches_generated` (regeneration must be a no-op).
Defaults to the primary run's file, `results_run2b.json`:

    uv run python scripts/render_readme_header.py [--results results.json]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results_run2b.json"
FALLBACK_RESULTS = "results.json"


def _resolve(path: str | None) -> Path:
    if path:
        return REPO_ROOT / path
    primary = REPO_ROOT / DEFAULT_RESULTS
    return primary if primary.exists() else REPO_ROOT / FALLBACK_RESULTS


def render(results_path: str | None = None, csv_path: str | None = None) -> str:
    target = _resolve(results_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    view = payload["views"]["primary"]
    gates = payload["validity_gates"]
    n_gates = len(gates.get("gates", []))
    gate_state = (
        f"ALL {n_gates} VALIDITY GATES PASS"
        if gates.get("all_passed")
        else f"VALIDITY FAILED: {', '.join(gates.get('failed', []))}"
    )
    if csv_path:
        csv_file = REPO_ROOT / csv_path
    else:
        csv_file = REPO_ROOT / (
            "data/human_validation_sample_run2b.csv"
            if target.stem != "results"
            else "data/human_validation_sample.csv"
        )
    labelled = 0
    total = 0
    if csv_file.exists():
        with csv_file.open(encoding="utf-8") as fh:
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    print(render(args.results, args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
