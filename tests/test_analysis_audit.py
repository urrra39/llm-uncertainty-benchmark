"""Tests for the post-hoc audit of the committed results.

Split in two. The first half is unit tests on synthetic input, where the right
answer is known analytically. The second half asserts against the committed
results.json, so the published claims about duplicate signals and per-dataset
intervals cannot drift away from the file they are drawn from.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from unc_bench.analysis.audit import (
    CI_METHOD_NOTE,
    SUSPECTED_DUPLICATE_PAIRS,
    AnalyticCI,
    analytic_auroc_ci,
    check_pairs,
    deduplicated_holm,
    holm,
    independent_signal_count,
    load_results,
    method_agreement,
    per_dataset_analytic_cis,
    render_report,
    spearman_between,
)
from unc_bench.analysis.metrics import auroc

RESULTS = Path("results.json")


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return load_results(RESULTS)


# --------------------------------------------------------------- Holm step-down


def test_holm_on_a_known_family() -> None:
    """Hand-checked: m=4, so adjusted values are 4p, 3p, 2p, p with a running max."""
    out = holm({"a": 0.001, "b": 0.01, "c": 0.02, "d": 0.5})
    assert out["a"][0] == pytest.approx(0.004)
    assert out["b"][0] == pytest.approx(0.03)
    assert out["c"][0] == pytest.approx(0.04)
    assert out["d"][0] == pytest.approx(0.5)
    assert [out[k][1] for k in "abcd"] == [True, True, True, False]


def test_holm_enforces_monotonicity() -> None:
    """A later hypothesis cannot end below an earlier one after adjustment."""
    out = holm({"a": 0.02, "b": 0.021, "c": 0.022})
    adjusted = [out[k][0] for k in "abc"]
    assert adjusted == sorted(adjusted)


def test_holm_clamps_at_one() -> None:
    assert holm({"a": 0.9, "b": 0.95})["b"][0] == 1.0


def test_holm_on_a_single_hypothesis_is_unadjusted() -> None:
    adjusted, significant = holm({"only": 0.04})["only"]
    assert adjusted == pytest.approx(0.04)
    assert significant


def test_smaller_family_never_produces_a_larger_adjusted_p() -> None:
    """The direction of the bias the README claims, asserted rather than stated."""
    full = {f"s{i}": 0.001 * (i + 1) for i in range(20)}
    reduced = {k: v for k, v in full.items() if k not in {"s0", "s1", "s2"}}
    adjusted_full = holm(full)
    adjusted_reduced = holm(reduced)
    for name in reduced:
        assert adjusted_reduced[name][0] <= adjusted_full[name][0] + 1e-12


# ------------------------------------------------------- analytic AUROC interval


def test_analytic_interval_brackets_the_point_estimate() -> None:
    low, high = analytic_auroc_ci(0.70, 40, 60)
    assert low < 0.70 < high


def test_analytic_interval_is_symmetric_about_the_point() -> None:
    point = 0.65
    low, high = analytic_auroc_ci(point, 36, 54)
    assert point - low == pytest.approx(high - point)


def test_analytic_interval_narrows_as_n_grows() -> None:
    narrow = analytic_auroc_ci(0.70, 400, 600)
    wide = analytic_auroc_ci(0.70, 40, 60)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_analytic_interval_mirrors_below_chance() -> None:
    """A sub-chance AUROC is a chance signal read backwards; widths must match."""
    above = analytic_auroc_ci(0.63, 36, 54)
    below = analytic_auroc_ci(0.37, 36, 54)
    assert (above[1] - above[0]) == pytest.approx(below[1] - below[0])
    assert below[0] == pytest.approx(1.0 - above[1])
    assert below[1] == pytest.approx(1.0 - above[0])


def test_analytic_interval_is_nan_on_degenerate_input() -> None:
    assert all(math.isnan(v) for v in analytic_auroc_ci(float("nan"), 10, 10))
    assert all(math.isnan(v) for v in analytic_auroc_ci(0.7, 0, 10))
    assert all(math.isnan(v) for v in analytic_auroc_ci(0.7, 10, 0))


def test_analytic_interval_agrees_with_a_bootstrap_on_clean_data() -> None:
    """Calibration check on untied Gaussian scores, where the formula's
    assumptions hold. Agreement here is what licenses using it at all; the
    committed data violates the no-ties assumption, which is why every caller
    labels the result as an approximation."""
    rng = np.random.default_rng(12345)
    n_pos, n_neg = 200, 300
    pos = rng.normal(0.8, 1.0, n_pos)
    neg = rng.normal(0.0, 1.0, n_neg)
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    point = auroc(scores, labels)

    draws = []
    for _ in range(2000):
        pi = rng.integers(0, n_pos, n_pos)
        ni = rng.integers(0, n_neg, n_neg)
        draws.append(
            auroc(
                np.concatenate([pos[pi], neg[ni]]),
                np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)]),
            )
        )
    boot_low, boot_high = np.percentile(draws, [2.5, 97.5])
    an_low, an_high = analytic_auroc_ci(point, n_pos, n_neg)
    assert abs(an_low - float(boot_low)) < 0.03
    assert abs(an_high - float(boot_high)) < 0.03


def test_analytic_ci_flags_an_out_of_range_endpoint() -> None:
    """A symmetric normal interval can leave [0, 1]; AUROC cannot."""
    item = AnalyticCI("x", 0.95, 0.88, 1.04, 27, 3)
    assert item.out_of_range
    assert not AnalyticCI("y", 0.60, 0.50, 0.70, 36, 54).out_of_range


def test_analytic_ci_chance_exclusion_and_margin() -> None:
    clears = AnalyticCI("a", 0.63, 0.507, 0.746, 36, 54)
    assert clears.excludes_chance
    assert clears.margin_from_chance == pytest.approx(0.007, abs=1e-3)

    spans = AnalyticCI("b", 0.55, 0.45, 0.65, 36, 54)
    assert not spans.excludes_chance
    assert spans.margin_from_chance < 0

    below = AnalyticCI("c", 0.30, 0.20, 0.40, 36, 54)
    assert below.excludes_chance


def test_selection_adjusted_p_exceeds_the_unadjusted_one() -> None:
    item = AnalyticCI("a", 0.6263, 0.5067, 0.7459, 36, 54)
    raw = item.chance_p_value()
    assert 0.0 < raw < 0.05
    assert item.selection_adjusted_p(18) > raw
    assert item.selection_adjusted_p(1) == pytest.approx(raw)


def test_selection_adjusted_p_clamps_at_one() -> None:
    assert AnalyticCI("a", 0.52, 0.40, 0.64, 36, 54).selection_adjusted_p(100) == 1.0


# ------------------------------------------------- against the committed results


def test_every_suspected_pair_is_measured_rank_equivalent(payload: dict[str, Any]) -> None:
    """The README's duplicate claim. Exactly +1.000, not merely close."""
    for pair in check_pairs(payload, SUSPECTED_DUPLICATE_PAIRS):
        assert pair.spearman == 1.0, f"{pair.a}/{pair.b} measured {pair.spearman!r}"
        assert pair.rank_equivalent


def test_a_non_duplicate_pair_is_not_rank_equivalent(payload: dict[str, Any]) -> None:
    """Guards the test above from passing because everything correlates at 1.0."""
    rho = spearman_between(payload, "a_mean_logprob", "b_distinct_count")
    assert abs(rho) < 1.0


def test_spearman_lookup_rejects_an_unknown_signal(payload: dict[str, Any]) -> None:
    with pytest.raises(KeyError):
        spearman_between(payload, "a_mean_logprob", "not_a_signal")


def test_independent_signal_count_matches_the_readme(payload: dict[str, Any]) -> None:
    total, independent = independent_signal_count(payload)
    assert (total, independent) == (21, 18)


def test_deduplicated_holm_reproduces_the_stored_adjustment(payload: dict[str, Any]) -> None:
    """The full-family column must equal what results.json already published,
    which is what makes the deduplicated column trustworthy."""
    stored = {
        c["name"]: (c["p_value_holm"], c["significant_holm"])
        for c in payload["views"]["primary"]["significance"]["comparisons"]
    }
    for row in deduplicated_holm(payload):
        expected_p, expected_sig = stored[row.name]
        assert row.holm_full == pytest.approx(expected_p)
        assert row.significant_full is expected_sig


def test_deduplication_drops_exactly_the_three_duplicates(payload: dict[str, Any]) -> None:
    rows = deduplicated_holm(payload)
    dropped = {r.name for r in rows if r.dropped}
    assert dropped == {"a_perplexity", "b_semantic_entropy_normalized", "b_distinct_fraction"}


def test_deduplication_never_loosens_an_adjusted_p(payload: dict[str, Any]) -> None:
    for row in deduplicated_holm(payload):
        if row.holm_dedup is not None:
            assert row.holm_dedup <= row.holm_full + 1e-12


def test_deduplication_changes_no_significance_verdict(payload: dict[str, Any]) -> None:
    """The published significance count survives deduplication. If this ever
    fails, the README's claim that the correction is merely conservative is
    wrong and the numbers have to move."""
    rows = deduplicated_holm(payload)
    assert [r.name for r in rows if r.verdict_changed] == []
    kept = [r for r in rows if not r.dropped]
    assert sum(1 for r in kept if r.significant_dedup) == 4
    assert sum(1 for r in rows if r.significant_full) == 4


def test_stored_significant_count_is_four(payload: dict[str, Any]) -> None:
    assert payload["views"]["primary"]["significance"]["n_significant_after_holm"] == 4
    assert payload["views"]["primary"]["significance"]["n_comparisons"] == 20


def test_popqa_class_counts_are_the_published_ones(payload: dict[str, Any]) -> None:
    cis = per_dataset_analytic_cis(payload, "popqa")
    assert len(cis) == 21
    assert {c.n_pos for c in cis} == {36}
    assert {c.n_neg for c in cis} == {54}


def test_exactly_one_popqa_interval_excludes_chance(payload: dict[str, Any]) -> None:
    """Defect 1's answer. One signal clears 0.50 and it is c_p_true_plain."""
    cis = per_dataset_analytic_cis(payload, "popqa")
    clearing = [c.name for c in cis if c.excludes_chance]
    assert clearing == ["c_p_true_plain"]


def test_the_popqa_exclusion_is_inside_the_method_error(payload: dict[str, Any]) -> None:
    """Why the README calls it inconclusive rather than shippable: the margin is
    smaller than the analytic method's own disagreement with the bootstrap."""
    cis = per_dataset_analytic_cis(payload, "popqa")
    top = next(c for c in cis if c.name == "c_p_true_plain")
    agreement = method_agreement(payload)
    assert 0 < top.margin_from_chance < agreement.max_endpoint_gap


def test_the_popqa_exclusion_does_not_survive_multiplicity(payload: dict[str, Any]) -> None:
    cis = per_dataset_analytic_cis(payload, "popqa")
    top = next(c for c in cis if c.name == "c_p_true_plain")
    _, independent = independent_signal_count(payload)
    assert top.chance_p_value() < 0.05
    assert top.selection_adjusted_p(independent) > 0.05


def test_mean_logprob_is_at_chance_on_popqa(payload: dict[str, Any]) -> None:
    """The signal the superseded recommendation named."""
    cis = per_dataset_analytic_cis(payload, "popqa")
    item = next(c for c in cis if c.name == "a_mean_logprob")
    assert item.auroc == pytest.approx(0.514, abs=0.001)
    assert not item.excludes_chance
    assert item.ci_low < 0.5 < item.ci_high


def test_triviaqa_intervals_are_unusable(payload: dict[str, Any]) -> None:
    """3 correct rows out of 30. The top intervals run past 1.0, which is the
    arithmetic signalling that the subset cannot support an interval."""
    cis = per_dataset_analytic_cis(payload, "triviaqa")
    assert cis[0].n_neg == 3
    assert any(c.out_of_range for c in cis)


def test_method_agreement_finds_no_chance_verdict_disagreement(payload: dict[str, Any]) -> None:
    agreement = method_agreement(payload)
    assert agreement.n_signals == 21
    assert agreement.chance_verdict_disagreements == 0
    assert agreement.max_endpoint_gap < 0.05


def test_report_names_its_method_and_the_inconclusive_verdict(payload: dict[str, Any]) -> None:
    text = render_report(payload)
    assert CI_METHOD_NOTE in text
    assert "INCONCLUSIVE" in text
    assert "21 table rows, 18 distinct orderings" in text
    assert "verdicts changed by deduplication: none" in text


def test_report_runs_on_the_secondary_view(payload: dict[str, Any]) -> None:
    assert render_report(payload, view="with_abstentions")


def test_load_results_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "r.json"
    target.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert load_results(target) == {"k": 1}


# ----------------------------------------------------------------- CLI wiring


def test_both_subcommands_are_registered() -> None:
    from unc_bench.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions and actions[0].choices is not None
    assert {"audit", "human-agreement"} <= set(actions[0].choices)


def test_audit_subcommand_prints_the_report(capsys: pytest.CaptureFixture[str]) -> None:
    from unc_bench.cli import main

    assert main(["audit", "--config", "configs/run2.yaml"]) == 0
    out = capsys.readouterr().out
    assert "Rank equivalence" in out
    assert "21 table rows, 18 distinct orderings" in out


def test_human_agreement_subcommand_exits_zero_on_an_unlabelled_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unlabelled file is the shipped state, not a failure."""
    from unc_bench.cli import main

    assert main(["human-agreement", "--config", "configs/run2.yaml"]) == 0
    assert "No human labels present" in capsys.readouterr().out


def test_human_agreement_subcommand_reports_a_missing_file(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from unc_bench.cli import main

    missing = tmp_path / "nope.csv"
    assert main(["human-agreement", "--config", "configs/run2.yaml", "--csv", str(missing)]) == 2
    assert "no validation CSV" in capsys.readouterr().err
