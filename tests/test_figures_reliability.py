"""Tests for the reliability-diagram figure (D5).

These assert the *contract* the figure has with `results.json`: it draws only
the probability-valued signals, it reads the before- and after-Platt ECEs that
the report already computed rather than deriving its own, it never invents a
point for an empty bin, and it raises rather than emitting an empty axes when
there is nothing to draw.

Nothing here checks pixels. The figure is verified through the data it selects
and the labels it builds, which is where a regression would actually hide.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from unc_bench.analysis.figures import (
    NothingToPlotError,
    _reliability_points,
    load_results,
    plot_reliability,
    reliability_panels,
)
from unc_bench.analysis.report import PROBABILITY_VALUED


def _calibration(
    *, counts: list[int], confidence: list[float | None], accuracy: list[float | None]
) -> dict[str, Any]:
    return {
        "bins": len(counts),
        "bin_counts": counts,
        "bin_confidence": confidence,
        "bin_accuracy": accuracy,
        "ece": 0.25,
    }


def _signal(
    *,
    family: str = "C",
    probability_valued: bool,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    auroc_point: float = 0.6,
) -> dict[str, Any]:
    return {
        "family": family,
        "auroc": {"point": auroc_point},
        "is_probability_valued": probability_valued,
        "calibration_before_platt": before,
        "calibration": after,
        "ece_before_platt": 0.4 if before else None,
        "ece_after_platt": 0.1 if after else None,
    }


def _results(signals: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": "unit",
        "analysis_config": {"ece_bins": 10},
        "views": {
            "primary": {
                "n": 120,
                "base_rate_incorrect": 0.525,
                "signals": signals,
                "ranking": sorted(signals),
            }
        },
    }


_POPULATED = _calibration(
    counts=[4, 0, 6],
    confidence=[0.1, None, 0.9],
    accuracy=[0.25, None, 0.5],
)


# --------------------------------------------------------- bin selection


def test_empty_bins_are_dropped_not_interpolated() -> None:
    # Three bins, the middle one empty. Two points must come back, and the
    # dropped bin must not reappear as an interpolated midpoint.
    conf, acc, counts = _reliability_points(_POPULATED)
    assert conf.tolist() == [0.1, 0.9]
    assert acc.tolist() == [0.25, 0.5]
    assert counts.tolist() == [4, 6]


def test_a_bin_with_a_count_but_a_null_accuracy_is_dropped() -> None:
    # Defensive: a non-finite accuracy alongside a positive count would plot as
    # a gap in the line at best and crash the scatter at worst.
    conf, acc, counts = _reliability_points(
        _calibration(counts=[3, 5], confidence=[0.2, 0.8], accuracy=[None, 0.6])
    )
    assert conf.tolist() == [0.8]
    assert acc.tolist() == [0.6]
    assert counts.tolist() == [5]


def test_missing_calibration_block_yields_no_points() -> None:
    # A signal the report marked probability-valued but for which no row landed
    # in the eval split stores null, and must not raise here.
    empty_blocks: list[dict[str, Any] | None] = [None, {}]
    for value in empty_blocks:
        conf, acc, counts = _reliability_points(value)
        assert conf.size == 0
        assert acc.size == 0
        assert counts.size == 0


def test_all_bins_empty_yields_no_points() -> None:
    conf, _, _ = _reliability_points(
        _calibration(counts=[0, 0], confidence=[None, None], accuracy=[None, None])
    )
    assert conf.size == 0


# ------------------------------------------------------------- plotting


def test_only_probability_valued_signals_are_drawn(tmp_path: Path) -> None:
    # A logprob signal has no meaningful position on a probability axis. If it
    # ever appears in this figure, someone has min-max scaled a logprob and is
    # plotting the calibration of the scaling.
    results = _results(
        {
            "c_p_true_plain": _signal(probability_valued=True, before=_POPULATED, after=_POPULATED),
            "a_mean_logprob": _signal(
                family="A", probability_valued=False, before=None, after=_POPULATED
            ),
        }
    )
    assert reliability_panels(results) == ["c_p_true_plain"]
    assert plot_reliability(results, tmp_path / "reliability.png").exists()


def test_a_signal_with_no_populated_bin_at_all_is_skipped() -> None:
    empty = _calibration(counts=[0, 0], confidence=[None, None], accuracy=[None, None])
    results = _results(
        {
            "c_p_true_plain": _signal(probability_valued=True, before=_POPULATED, after=_POPULATED),
            "c_verbal_confidence": _signal(probability_valued=True, before=empty, after=empty),
        }
    )
    assert reliability_panels(results) == ["c_p_true_plain"]


def test_panels_are_ordered_deterministically() -> None:
    # The panel order must not depend on dict insertion order, or the figure
    # silently reshuffles between runs and a reader comparing two copies of it
    # concludes something moved.
    entry = _signal(probability_valued=True, before=_POPULATED, after=_POPULATED)
    forward = _results({"c_p_true_plain": entry, "c_verbal_confidence": entry})
    reverse = _results({"c_verbal_confidence": entry, "c_p_true_plain": entry})
    assert reliability_panels(forward) == reliability_panels(reverse)
    assert reliability_panels(forward) == ["c_p_true_plain", "c_verbal_confidence"]


def test_nothing_to_plot_rather_than_an_empty_axes(tmp_path: Path) -> None:
    # `render_all` catches this and carries on with the other figures. Drawing
    # a bare diagonal with no curve on it would look like a result.
    results = _results(
        {"a_mean_logprob": _signal(family="A", probability_valued=False, before=None, after=None)}
    )
    with pytest.raises(NothingToPlotError):
        plot_reliability(results, tmp_path / "reliability.png")


def test_half_a_pair_still_draws(tmp_path: Path) -> None:
    # The after-Platt curve alone is a legitimate figure: it means no row landed
    # in a pre-Platt bin, not that the signal is undrawable.
    results = _results(
        {"c_verbal_confidence": _signal(probability_valued=True, before=None, after=_POPULATED)}
    )
    assert plot_reliability(results, tmp_path / "reliability.png").exists()


# ------------------------------------------- against the committed results.json


@pytest.fixture
def committed_results() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "results.json"
    if not path.exists():  # pragma: no cover - results.json is committed
        pytest.skip("results.json is not present")
    return load_results(path)


def test_committed_results_have_exactly_three_probability_valued_signals(
    committed_results: dict[str, Any],
) -> None:
    signals = committed_results["views"]["primary"]["signals"]
    flagged = {name for name, entry in signals.items() if entry["is_probability_valued"]}
    assert flagged == set(PROBABILITY_VALUED)


def test_committed_results_carry_both_eces_for_every_probability_signal(
    committed_results: dict[str, Any],
) -> None:
    # The figure's legend reads these two fields directly. If either is null the
    # caption would say "ECE nan", which is the stale-figure failure this whole
    # figure exists to fix.
    signals = committed_results["views"]["primary"]["signals"]
    for name in PROBABILITY_VALUED:
        entry = signals[name]
        assert entry["ece_before_platt"] is not None, name
        assert entry["ece_after_platt"] is not None, name
        assert entry["calibration_before_platt"] is not None, name
        assert entry["calibration"] is not None, name


def test_figure_renders_from_the_committed_results(
    committed_results: dict[str, Any], tmp_path: Path
) -> None:
    out = plot_reliability(committed_results, tmp_path / "reliability.png")
    assert out.stat().st_size > 0


def test_no_stale_calibration_figure_is_committed() -> None:
    # The old calibration.png predated the before/after-Platt fields and was
    # therefore stale. It was deleted rather than left to be misread.
    figures = Path(__file__).resolve().parents[1] / "figures"
    assert not (figures / "calibration.png").exists()


def _readme_figure_references() -> set[str]:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    return set(re.findall(r"figures/[A-Za-z0-9_.-]+\.png", readme))


def test_every_figure_referenced_by_the_readme_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    referenced = _readme_figure_references()
    assert referenced, "the README references no figure at all"
    for name in sorted(referenced):
        assert (root / name).exists(), f"README references a missing figure: {name}"


def test_no_figure_on_disk_is_orphaned() -> None:
    # The mirror of the test above: a .png nothing points at is dead weight and
    # the next reader cannot tell whether it is current.
    root = Path(__file__).resolve().parents[1]
    referenced = _readme_figure_references()
    for png in sorted((root / "figures").glob("*.png")):
        assert f"figures/{png.name}" in referenced, f"orphaned figure: {png.name}"


def test_results_json_parses_and_is_the_published_run() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert data["validity_gates"]["all_passed"] is True
    assert data["validity_gates"]["ranking_publishable"] is True
    assert data["views"]["primary"]["n"] == 120
