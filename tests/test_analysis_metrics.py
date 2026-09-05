"""Metric tests. Every target is hand-computed, not recorded from an output.

The point of a test that asserts `auroc(...) == 0.75` is that 0.75 was derived
independently. Where a value comes from an enumeration I have written the
enumeration into the comment so the assertion can be checked without rerunning
anything.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from unc_bench.analysis.metrics import (
    apply_platt,
    auroc,
    bootstrap_auroc_ci,
    expected_calibration_error,
    fit_platt,
    risk_coverage,
    spearman_matrix,
    usable_mask,
)


def _f(values: list[float]) -> npt.NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _b(values: list[bool]) -> npt.NDArray[np.bool_]:
    return np.array(values, dtype=bool)


# ------------------------------------------------------------------- AUROC


def test_auroc_perfect_separation() -> None:
    # Every positive scores above every negative: all 4 pairs concordant.
    scores = _f([0.1, 0.2, 0.8, 0.9])
    labels = _b([False, False, True, True])
    assert auroc(scores, labels) == 1.0


def test_auroc_perfect_inversion_is_zero() -> None:
    # The mirror image. A signal at 0.0 is exactly as informative as one at 1.0
    # and is reported as 0.0 rather than folded, because the fold would hide an
    # orientation bug.
    scores = _f([0.9, 0.8, 0.2, 0.1])
    labels = _b([False, False, True, True])
    assert auroc(scores, labels) == 0.0


def test_auroc_hand_computed_with_one_discordant_pair() -> None:
    # positives {0.5, 0.9}, negatives {0.1, 0.7}.
    # Pairs: (0.5>0.1) yes, (0.5>0.7) no, (0.9>0.1) yes, (0.9>0.7) yes -> 3/4.
    scores = _f([0.1, 0.5, 0.7, 0.9])
    labels = _b([False, True, False, True])
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_auroc_counts_a_tie_as_half() -> None:
    # positives {0.5}, negatives {0.5}. The single pair is tied, worth 0.5.
    assert auroc(_f([0.5, 0.5]), _b([True, False])) == pytest.approx(0.5)


def test_auroc_with_ties_is_independent_of_row_order() -> None:
    # The reason for midranks. `t_answer_length` takes a handful of distinct
    # values over the whole study, so ties are most of the data, and a
    # tie-breaking-by-input-order implementation would give two different
    # numbers for the same data depending on how it arrived.
    scores = _f([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    labels = _b([True, False, True, False, True, False])
    first = auroc(scores, labels)
    order = np.array([4, 0, 5, 2, 1, 3])
    second = auroc(scores[order], labels[order])
    assert first == pytest.approx(second)


def test_auroc_is_nan_when_one_class_is_absent() -> None:
    # Not 0.5. There is no coin flip here; there is nothing to measure.
    assert math.isnan(auroc(_f([0.1, 0.2, 0.3]), _b([True, True, True])))
    assert math.isnan(auroc(_f([0.1, 0.2, 0.3]), _b([False, False, False])))


def test_auroc_ignores_nan_rows_but_keeps_the_rest() -> None:
    # The NaN row is family B's "not measured" marker. Dropping it must not
    # change the AUROC of the rows that were measured.
    scores = _f([0.1, 0.5, 0.7, 0.9, float("nan")])
    labels = _b([False, True, False, True, True])
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_auroc_of_a_constant_signal_is_one_half() -> None:
    # Every pair is tied. A signal that says the same thing about every item
    # carries no information, and 0.5 is the correct reading.
    scores = _f([2.0] * 6)
    labels = _b([True, False, True, False, True, False])
    assert auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="paired arrays"):
        auroc(_f([0.1, 0.2]), _b([True]))


# --------------------------------------------------------------- bootstrap


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(7)
    scores = _f(list(rng.normal(0, 1, 60)) + list(rng.normal(1.5, 1, 60)))
    labels = _b([False] * 60 + [True] * 60)
    ci = bootstrap_auroc_ci(scores, labels, resamples=400, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert 0.0 <= ci.low <= 1.0
    assert 0.0 <= ci.high <= 1.0
    assert ci.width > 0.0


def test_bootstrap_ci_is_reproducible_from_the_seed() -> None:
    scores = _f([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.5, 0.7])
    labels = _b([False, True, False, True, False, True, False, True])
    a = bootstrap_auroc_ci(scores, labels, resamples=200, seed=42)
    b = bootstrap_auroc_ci(scores, labels, resamples=200, seed=42)
    assert (a.low, a.point, a.high) == (b.low, b.point, b.high)


def test_bootstrap_ci_never_produces_a_degenerate_resample() -> None:
    # Stratification is what guarantees this. With 3 positives out of 20 an
    # unstratified bootstrap draws an all-negative resample often enough to
    # matter, and its AUROC would be undefined.
    scores = _f([float(i) for i in range(20)])
    labels = _b([i in (17, 18, 19) for i in range(20)])
    ci = bootstrap_auroc_ci(scores, labels, resamples=500, seed=3)
    assert ci.resamples == 500
    assert not math.isnan(ci.low)


def test_bootstrap_ci_of_a_random_signal_contains_one_half() -> None:
    # The null check the trivial baseline exists for. If this fails the
    # bootstrap is wrong, not the coin.
    rng = np.random.default_rng(11)
    scores = _f(list(rng.random(200)))
    labels = _b([bool(v) for v in rng.integers(0, 2, 200)])
    ci = bootstrap_auroc_ci(scores, labels, resamples=800, seed=5)
    assert ci.low <= 0.5 <= ci.high


def test_bootstrap_ci_is_nan_without_both_classes() -> None:
    ci = bootstrap_auroc_ci(_f([1.0, 2.0]), _b([True, True]), resamples=100, seed=1)
    assert math.isnan(ci.low) and math.isnan(ci.high)
    assert ci.resamples == 0


# ----------------------------------------------------------- risk-coverage


def test_risk_coverage_of_a_perfect_signal() -> None:
    # 2 correct then 2 incorrect, ranked correctly. Risk at k=1,2,3,4 is
    # 0, 0, 1/3, 2/4 -> AURC = (0 + 0 + 1/3 + 1/2) / 4 = 0.208333...
    scores = _f([0.1, 0.2, 0.8, 0.9])
    labels = _b([False, False, True, True])
    rc = risk_coverage(scores, labels)
    assert rc.risk[:2] == [0.0, 0.0]
    assert rc.risk[2] == pytest.approx(1 / 3)
    assert rc.risk[3] == pytest.approx(0.5)
    assert rc.aurc == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert rc.base_rate == pytest.approx(0.5)


def test_risk_coverage_ends_at_the_base_rate() -> None:
    # Full coverage answers everything, so the last risk value IS the base rate
    # by definition. A curve that does not land there is computing something
    # other than what it claims.
    rng = np.random.default_rng(2)
    scores = _f(list(rng.random(50)))
    labels = _b([bool(v) for v in rng.integers(0, 2, 50)])
    rc = risk_coverage(scores, labels)
    assert rc.risk[-1] == pytest.approx(rc.base_rate)
    assert rc.coverage[-1] == pytest.approx(1.0)


def test_coverage_at_target_is_none_when_the_target_is_unreachable() -> None:
    # Every item is wrong, so no prefix has risk <= 0.1. None, not 0.0: a
    # coverage of zero would suggest a valid operating point exists.
    scores = _f([0.1, 0.2, 0.3])
    labels = _b([True, True, True])
    rc = risk_coverage(scores, labels, target_accuracy=0.9)
    assert rc.coverage_at_target is None


def test_coverage_at_target_takes_the_largest_qualifying_prefix() -> None:
    # 8 correct then 2 wrong, perfectly ranked. At 90% target accuracy the
    # threshold is risk <= 0.1: k=9 gives 1/9 = 0.111 which fails, k=8 gives 0.
    # But k=10 gives 0.2. The largest qualifying prefix is k=8 -> coverage 0.8.
    scores = _f([float(i) for i in range(10)])
    labels = _b([i >= 8 for i in range(10)])
    rc = risk_coverage(scores, labels, target_accuracy=0.9)
    assert rc.coverage_at_target == pytest.approx(0.8)


# ------------------------------------------------------------- calibration


def test_ece_is_zero_for_a_perfectly_calibrated_signal() -> None:
    # Two bins, each internally consistent: 10 items at p=0.05 of which 0 are
    # wrong is not exact, so use p=0.0 and p=1.0 where the match is exact.
    probs = _f([0.0] * 10 + [1.0] * 10)
    outcomes = _b([False] * 10 + [True] * 10)
    cal = expected_calibration_error(probs, outcomes, bins=10)
    assert cal.ece == pytest.approx(0.0)


def test_ece_of_a_maximally_wrong_signal_is_one() -> None:
    # Predicts 0 for every item and every item is wrong.
    cal = expected_calibration_error(_f([0.0] * 8), _b([True] * 8), bins=10)
    assert cal.ece == pytest.approx(1.0)


def test_ece_is_hand_computable_on_two_bins() -> None:
    # 4 items at p=0.1, 1 wrong -> |0.25 - 0.1| = 0.15, weight 4/8
    # 4 items at p=0.9, 3 wrong -> |0.75 - 0.9| = 0.15, weight 4/8
    # ECE = 0.5 * 0.15 + 0.5 * 0.15 = 0.15
    probs = _f([0.1] * 4 + [0.9] * 4)
    outcomes = _b([True, False, False, False, True, True, True, False])
    cal = expected_calibration_error(probs, outcomes, bins=10)
    assert cal.ece == pytest.approx(0.15)


def test_empty_bins_are_reported_not_interpolated() -> None:
    # At n=150 over 10 bins several bins hold nothing. Drawing a reliability
    # curve through invented points there would be the most misleading thing in
    # the figure set, so empty bins carry count 0 and NaN.
    cal = expected_calibration_error(_f([0.05, 0.95]), _b([False, True]), bins=10)
    assert cal.bin_counts[0] == 1
    assert cal.bin_counts[-1] == 1
    assert cal.bin_counts[4] == 0
    assert math.isnan(cal.bin_accuracy[4])


def test_probability_of_exactly_one_lands_in_the_last_bin() -> None:
    # An off-by-one here silently drops every p == 1.0 row, which after Platt
    # scaling with a steep slope is a real population.
    cal = expected_calibration_error(_f([1.0, 1.0]), _b([True, True]), bins=10)
    assert cal.bin_counts[-1] == 2
    assert sum(cal.bin_counts) == 2


# ------------------------------------------------------------------ Platt


def test_platt_recovers_the_direction_of_the_relationship() -> None:
    rng = np.random.default_rng(5)
    scores = _f(list(rng.normal(0, 1, 200)))
    labels = _b([bool(v) for v in (scores > 0)])
    a, b = fit_platt(scores, labels)
    # Higher score means more likely wrong, so the slope must be positive.
    assert a > 0
    probs = apply_platt(scores, a, b)
    assert float(np.mean(probs[scores > 1.0])) > float(np.mean(probs[scores < -1.0]))


def test_platt_survives_an_enormous_input_scale() -> None:
    # Clamped perplexity reaches ~5e8. Newton's method on an unstandardized
    # feature of that magnitude overflows on the first step, which is why the
    # fit standardizes internally and folds the scale back in afterwards.
    rng = np.random.default_rng(9)
    base = _f(list(rng.random(100)))
    scores = base * 5e8
    labels = _b([bool(v) for v in (base > 0.5)])
    a, b = fit_platt(scores, labels)
    assert math.isfinite(a) and math.isfinite(b)
    probs = apply_platt(scores, a, b)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_platt_on_a_constant_signal_returns_the_base_rate() -> None:
    # No slope is identifiable. The best available map is the base rate, and it
    # must be a probability rather than a NaN that poisons the ECE.
    scores = _f([3.0] * 10)
    labels = _b([True] * 3 + [False] * 7)
    a, b = fit_platt(scores, labels)
    assert a == 0.0
    probs = apply_platt(scores, a, b)
    assert float(probs[0]) == pytest.approx(0.3, abs=1e-6)


def test_platt_preserves_nan_rows() -> None:
    probs = apply_platt(_f([1.0, float("nan"), 2.0]), 1.0, 0.0)
    assert math.isnan(float(probs[1]))
    assert math.isfinite(float(probs[0]))


# --------------------------------------------------------------- Spearman


def test_spearman_is_one_for_a_monotone_transform() -> None:
    # The specific case this matrix exists to expose: mean logprob and
    # perplexity are monotone transforms of each other, so a table listing both
    # as separate findings is listing one finding twice.
    a = _f([1.0, 2.0, 3.0, 4.0])
    b = _f([10.0, 20.0, 30.0, 40.0])
    names, matrix = spearman_matrix({"a": a, "b": b})
    assert names == ["a", "b"]
    assert matrix[0][1] == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_reversal() -> None:
    _, matrix = spearman_matrix({"a": _f([1.0, 2.0, 3.0, 4.0]), "b": _f([4.0, 3.0, 2.0, 1.0])})
    assert matrix[0][1] == pytest.approx(-1.0)


def test_spearman_diagonal_is_one() -> None:
    _, matrix = spearman_matrix({"a": _f([1.0, 5.0, 2.0, 4.0])})
    assert matrix[0][0] == pytest.approx(1.0)


def test_spearman_against_a_constant_column_is_nan() -> None:
    # Not 0.0. There is no monotone relationship defined against a column that
    # never varies, and 0.0 would read as a measured absence of one.
    _, matrix = spearman_matrix({"a": _f([1.0, 2.0, 3.0]), "b": _f([7.0, 7.0, 7.0])})
    assert math.isnan(matrix[0][1])


def test_spearman_uses_each_pairs_own_usable_overlap() -> None:
    # Family B is NaN wherever its pass did not reach. Ranking a column once
    # over its own usable set and then correlating two differently-masked rank
    # vectors compares ranks computed over different populations.
    nan = float("nan")
    a = _f([1.0, 2.0, 3.0, 4.0, 5.0])
    b = _f([nan, 2.0, 3.0, 4.0, nan])
    _, matrix = spearman_matrix({"a": a, "b": b})
    assert matrix[0][1] == pytest.approx(1.0)


def test_spearman_is_nan_with_too_few_shared_rows() -> None:
    nan = float("nan")
    _, matrix = spearman_matrix({"a": _f([1.0, 2.0, 3.0]), "b": _f([1.0, nan, nan])})
    assert math.isnan(matrix[0][1])


# ------------------------------------------------------------------- misc


def test_usable_mask_excludes_nan_and_infinity() -> None:
    mask = usable_mask(_f([1.0, float("nan"), float("inf"), -float("inf"), 0.0]))
    assert list(mask) == [True, False, False, False, True]


# ------------------------------------------------------- cluster bootstrap


def _ids(values: list[int]) -> npt.NDArray[np.int64]:
    return np.array(values, dtype=np.int64)


def test_singleton_clusters_reproduce_the_row_bootstrap_exactly() -> None:
    # One cluster per row: the cluster draw is the row draw given the same
    # seed, so both point and endpoints must match bit-for-bit.
    rng = np.random.default_rng(3)
    scores = _f(list(rng.normal(size=40)))
    labels = _b([True] * 20 + [False] * 20)
    singletons = _ids(list(range(40)))
    row = bootstrap_auroc_ci(scores, labels, resamples=500, seed=11)
    clustered = bootstrap_auroc_ci(scores, labels, resamples=500, seed=11, cluster_ids=singletons)
    assert (row.point, row.low, row.high) == (clustered.point, clustered.low, clustered.high)


def test_duplicated_rows_widen_the_interval_under_clustering() -> None:
    # Twenty unique items each appearing twice: the row bootstrap sees n=40
    # independent rows, the cluster bootstrap sees 20 pairs that always travel
    # together, so its interval must be wider.
    rng = np.random.default_rng(5)
    base = _f(list(rng.normal(size=20)))
    scores = _f(list(base) + list(base))
    labels = _b([True] * 10 + [False] * 10 + [True] * 10 + [False] * 10)
    pairs = _ids(list(range(20)) + list(range(20)))
    row = bootstrap_auroc_ci(scores, labels, resamples=1000, seed=11)
    clustered = bootstrap_auroc_ci(scores, labels, resamples=1000, seed=11, cluster_ids=pairs)
    assert clustered.width > row.width


def test_cluster_ids_must_align_with_usable_rows() -> None:
    scores = _f([0.1, 0.9, 0.2, 0.8])
    labels = _b([False, True, False, True])
    with pytest.raises(ValueError, match="one id per input row"):
        bootstrap_auroc_ci(scores, labels, resamples=50, seed=0, cluster_ids=_ids([0, 1]))


# ------------------------------------------------- distinct-signal ranking


def test_declared_duplicates_drop_on_monotone_data() -> None:
    from unc_bench.analysis.extended import resolve_distinct_ranking
    from unc_bench.signals import consistency, logprob_signals, trivial

    assert logprob_signals.PERPLEXITY.rank_equivalent_to == "a_mean_logprob"
    assert consistency.DISTINCT_FRACTION.rank_equivalent_to == "b_distinct_count"
    assert trivial.RANDOM.rank_equivalent_to is None

    mean = _f([-3.0, -2.0, -1.0, -0.5, -0.1])
    columns = {
        "a_mean_logprob": mean,
        "a_perplexity": _f(
            [math.exp(3.0), math.exp(2.0), math.exp(1.0), math.exp(0.5), math.exp(0.1)]
        ),
        "t_random": _f([0.5, 0.1, 0.9, 0.3, 0.7]),
    }
    distinct, dropped = resolve_distinct_ranking(
        ["a_mean_logprob", "a_perplexity", "t_random"], columns
    )
    assert distinct == ["a_mean_logprob", "t_random"]
    assert dropped == ["a_perplexity"]


def test_broken_equivalence_raises_loudly() -> None:
    from unc_bench.analysis.extended import resolve_distinct_ranking
    from unc_bench.signals.logprob_signals import PERPLEXITY

    assert PERPLEXITY.rank_equivalent_to == "a_mean_logprob"

    columns = {
        "a_mean_logprob": _f([-3.0, -2.0, -1.0, -0.5, -0.1]),
        # Genuinely different ranks (a reversal would still be |rho| == 1.0).
        "a_perplexity": _f([1.0, 5.0, 2.0, 4.0, 3.0]),
        "t_random": _f([0.5, 0.1, 0.9, 0.3, 0.7]),
    }
    with pytest.raises(AssertionError, match="rank_equivalent_to"):
        resolve_distinct_ranking(["a_mean_logprob", "a_perplexity", "t_random"], columns)
