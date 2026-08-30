"""Tests for the two-run AUROC comparison.

The load-bearing test is `test_self_comparison_reports_zero_differences`: the
committed results file compared against itself must report every signal
identical and zero differences. If that ever fails, the comparison is reading
something that is not stable within one file and no cross-run reading of it
would be trustworthy.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from unc_bench.analysis.compare import (
    compare_signals,
    count_differences,
    load,
    render,
    summarize,
)

RESULTS = Path("results.json")


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return load(RESULTS)


# ------------------------------------------------------- the self-comparison


def test_self_comparison_reports_zero_differences(payload: dict[str, Any]) -> None:
    """The committed run #2 file against itself: no signal may differ."""
    assert count_differences(payload, copy.deepcopy(payload)) == 0


def test_self_comparison_marks_every_signal_identical(payload: dict[str, Any]) -> None:
    rows = compare_signals(payload, copy.deepcopy(payload))
    assert rows, "the committed file has no signals to compare"
    assert all(row.identical for row in rows)
    assert all(row.both_present for row in rows)
    assert all(row.delta == 0.0 for row in rows)


def test_self_comparison_text_says_the_files_agree(payload: dict[str, Any]) -> None:
    text = render(payload, copy.deepcopy(payload))
    assert "differing               0" in text
    assert "agree on every signal" in text
    # Same file, so the row sets are identical and the header must say so
    # rather than warning about incomparable runs.
    assert "same row set" in text


def test_self_comparison_covers_every_committed_signal(payload: dict[str, Any]) -> None:
    rows = compare_signals(payload, copy.deepcopy(payload))
    assert len(rows) == len(payload["views"]["primary"]["signals"])


# ------------------------------------------------------------ real differences


def _bump(payload: dict[str, Any], signal: str, delta: float) -> dict[str, Any]:
    other = copy.deepcopy(payload)
    entry = other["views"]["primary"]["signals"][signal]["auroc"]
    entry["point"] = float(entry["point"]) + delta
    return other


def test_a_moved_point_estimate_is_counted(payload: dict[str, Any]) -> None:
    other = _bump(payload, "t_random", 0.10)
    assert count_differences(payload, other) == 1


def test_delta_is_right_minus_left(payload: dict[str, Any]) -> None:
    other = _bump(payload, "t_random", 0.10)
    row = next(r for r in compare_signals(payload, other) if r.name == "t_random")
    assert row.delta == pytest.approx(0.10)


def test_ordering_follows_the_left_run(payload: dict[str, Any]) -> None:
    rows = compare_signals(payload, copy.deepcopy(payload))
    points = [r.left_point for r in rows if r.in_left]
    assert points == sorted(points, reverse=True)


def test_a_signal_absent_from_one_side_is_named_not_dropped(
    payload: dict[str, Any],
) -> None:
    other = copy.deepcopy(payload)
    del other["views"]["primary"]["signals"]["t_random"]
    rows = compare_signals(payload, other)
    missing = next(r for r in rows if r.name == "t_random")
    assert missing.in_left and not missing.in_right
    assert not missing.both_present
    # Still reported, and reported as a difference.
    assert len(rows) == len(payload["views"]["primary"]["signals"])
    assert "absent from" in render(payload, other)


def test_differing_row_sets_are_flagged(payload: dict[str, Any]) -> None:
    other = copy.deepcopy(payload)
    other["frozen_analysis_set"]["primary"]["qid_digest"] = "0000000000000000"
    text = render(payload, other)
    assert "different row sets" in text


# ------------------------------------------------------------------- overlap


def test_disjoint_intervals_are_described_as_disjoint(payload: dict[str, Any]) -> None:
    other = copy.deepcopy(payload)
    entry = other["views"]["primary"]["signals"]["t_random"]["auroc"]
    entry["point"] = 0.99
    entry["ci_low"] = 0.95
    entry["ci_high"] = 1.0
    row = next(r for r in compare_signals(payload, other) if r.name == "t_random")
    assert row.intervals_overlap is False


def test_overlap_is_none_when_an_endpoint_is_missing(payload: dict[str, Any]) -> None:
    other = copy.deepcopy(payload)
    other["views"]["primary"]["signals"]["t_random"]["auroc"]["ci_low"] = None
    row = next(r for r in compare_signals(payload, other) if r.name == "t_random")
    assert row.intervals_overlap is None


def test_no_paired_test_is_claimed(payload: dict[str, Any]) -> None:
    """The comparison must not present itself as a significance test."""
    text = render(payload, copy.deepcopy(payload))
    assert "no paired test between them is available" in text


# -------------------------------------------------------------------- header


def test_summary_reads_the_committed_run(payload: dict[str, Any]) -> None:
    summary = summarize(payload, "run2")
    assert summary.n == payload["views"]["primary"]["n"]
    assert summary.n_incorrect == payload["views"]["primary"]["n_incorrect"]
    assert summary.qid_digest == payload["frozen_analysis_set"]["primary"]["qid_digest"]
    assert summary.model == payload["model_under_test"]["name"]


def test_null_values_render_as_a_dash(tmp_path: Path) -> None:
    """A results file storing null must not print as a measured zero."""
    minimal: dict[str, Any] = {
        "run_name": "x",
        "model_under_test": {"name": "m"},
        "dataset": {"mix": {"popqa": 1}},
        "analysis_config": {"bootstrap_resamples": 10},
        "validity_gates": {"all_passed": True},
        "frozen_analysis_set": {"primary": {"qid_digest": "d"}},
        "signal_catalog": {"s": {"family": "T"}},
        "views": {
            "primary": {
                "n": 1,
                "n_incorrect": 1,
                "n_correct": 0,
                "base_rate_incorrect": None,
                "signals": {"s": {"auroc": {"point": None, "ci_low": None, "ci_high": None}}},
            }
        },
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(minimal), encoding="utf-8")
    text = render(load(path), load(path))
    assert "-" in text
    # A null point estimate is not a difference from itself, but it is also not
    # "identical" in the sense the self-comparison test pins, because NaN != NaN.
    # What must hold is that rendering does not crash and prints no fake number.
    assert "0.000" not in text.split("Per-signal")[1].split("signals compared")[0]
