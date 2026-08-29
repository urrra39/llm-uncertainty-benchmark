"""Family A, checked against hand-computed values.

The entropy targets in here are arithmetic, not regression baselines: ln2 for a
uniform top-2, ln5 for a uniform top-5, 0 for a point mass, and 0.8018185... for
p=[0.7, 0.2, 0.1]. If the renormalization or the log base is wrong these fail
immediately rather than producing a plausible-looking column.
"""

from __future__ import annotations

import math

import pytest

from unc_bench.config import SignalsSpec
from unc_bench.signals.base import (
    FAMILY_A,
    ORIENT_CONFIDENCE,
    ORIENT_RISK,
    get_spec,
    orient_all,
    signal_names,
)
from unc_bench.signals.logprob_signals import (
    FAMILY_A_SIGNALS,
    FIRST_TOKEN_LOGPROB,
    FIRST_TOKEN_MARGIN,
    LENGTH_NORMALIZED_LOGPROB,
    MAX_TOP5_ENTROPY,
    MEAN_LOGPROB,
    MEAN_TOP5_ENTROPY,
    MIN_LOGPROB,
    PERPLEXITY,
    TOTAL_LOGPROB,
    compute_family_a,
    length_penalty,
    top1_top2_margin,
    truncated_entropy,
)
from unc_bench.types import Generation, TokenLogprob

SPEC = SignalsSpec()


def _token(probs: list[float], *, chosen: int = 0, token: str = "x") -> TokenLogprob:
    """A TokenLogprob whose top-k is exactly `probs`, in probability space."""
    top = {f"t{i}": math.log(p) for i, p in enumerate(probs)}
    keys = list(top)
    return TokenLogprob(
        token=keys[chosen] if token == "x" else token,
        logprob=top[keys[chosen]],
        top_logprobs=top,
    )


def _greedy(tokens: tuple[TokenLogprob, ...]) -> Generation:
    return Generation(
        text="answer",
        answer_token_logprobs=tokens,
        is_greedy=True,
        seed=0,
        temperature=0.0,
    )


# ---------------------------------------------------------------- entropy


def test_entropy_uniform_top2_is_ln2() -> None:
    assert truncated_entropy(_token([0.5, 0.5])) == pytest.approx(math.log(2))


def test_entropy_uniform_top5_is_ln5() -> None:
    assert truncated_entropy(_token([0.2] * 5)) == pytest.approx(math.log(5))


def test_entropy_point_mass_is_zero() -> None:
    assert truncated_entropy(_token([1.0])) == pytest.approx(0.0)


def test_entropy_matches_hand_computed_skewed_case() -> None:
    # -(0.7 ln0.7 + 0.2 ln0.2 + 0.1 ln0.1) = 0.8018185525433373
    got = truncated_entropy(_token([0.7, 0.2, 0.1]))
    assert got == pytest.approx(0.8018185525433373, abs=1e-9)


def test_entropy_renormalizes_a_truncated_top_k() -> None:
    """A top-5 that sums to 0.5 must still read as a distribution.

    Halving every probability leaves the renormalized distribution unchanged, so
    the entropy is identical to the un-truncated uniform case.
    """
    assert truncated_entropy(_token([0.1] * 5)) == pytest.approx(math.log(5))


def test_entropy_of_empty_top_k_is_zero_not_nan() -> None:
    """No observed alternatives means no evidence of uncertainty at that step.

    This is the one place 0.0 is right rather than NaN: the aggregate over
    positions still has to be computable when a backend reports no top-k.
    """
    bare = TokenLogprob(token="a", logprob=-0.5, top_logprobs={})
    assert truncated_entropy(bare) == 0.0


# ----------------------------------------------------------------- margin


def test_margin_is_probability_space_difference() -> None:
    assert top1_top2_margin(_token([0.7, 0.2, 0.1])) == pytest.approx(0.5)


def test_margin_of_uniform_top2_is_zero() -> None:
    assert top1_top2_margin(_token([0.5, 0.5])) == pytest.approx(0.0)


def test_margin_with_one_observed_alternative_uses_zero_runner_up() -> None:
    assert top1_top2_margin(_token([1.0])) == pytest.approx(1.0)


def test_margin_of_empty_top_k_is_nan() -> None:
    bare = TokenLogprob(token="a", logprob=-0.5, top_logprobs={})
    assert math.isnan(top1_top2_margin(bare))


# ------------------------------------------------------------- aggregates


def test_aggregates_match_hand_computed_values() -> None:
    tokens = (
        TokenLogprob("a", -0.10, {"a": -0.10, "b": -3.0}),
        TokenLogprob("b", -0.90, {"b": -0.90, "c": -1.1}),
        TokenLogprob("c", -2.00, {"c": -2.00, "d": -2.1}),
    )
    out = compute_family_a(_greedy(tokens), SPEC)
    total = -3.0
    assert out[TOTAL_LOGPROB.name] == pytest.approx(total)
    assert out[MEAN_LOGPROB.name] == pytest.approx(-1.0)
    assert out[MIN_LOGPROB.name] == pytest.approx(-2.0)
    assert out[FIRST_TOKEN_LOGPROB.name] == pytest.approx(-0.10)
    assert out[PERPLEXITY.name] == pytest.approx(math.e)
    penalty = ((5.0 + 3) / 6.0) ** SPEC.length_penalty_alpha
    assert out[LENGTH_NORMALIZED_LOGPROB.name] == pytest.approx(total / penalty)


def test_length_normalized_is_not_a_duplicate_of_the_mean() -> None:
    """The two must rank some pair differently, or one column is dead weight.

    My first attempt at this test asserted the wrong thing. I picked two spans
    with equal total logprob and different lengths, and both signals preferred
    the longer one, because dividing a fixed negative total by anything larger
    moves it the same direction. The pair that actually separates them has to
    differ in total as well: one token at -3.0 against six tokens at -1.0. The
    mean prefers the six-token span (-1.0 vs -3.0) while the Wu-penalized total
    prefers the single token (-3.0 vs -4.17), because the penalty grows
    sublinearly in n and cannot pay for six times the total. If no such pair
    existed the heatmap would read 1.00 and one column should be deleted.
    """
    short = _greedy((TokenLogprob("a", -3.0, {"a": -3.0}),))
    long = _greedy(tuple(TokenLogprob("a", -1.0, {"a": -1.0}) for _ in range(6)))
    a = compute_family_a(short, SPEC)
    b = compute_family_a(long, SPEC)
    assert a[MEAN_LOGPROB.name] < b[MEAN_LOGPROB.name]
    assert a[LENGTH_NORMALIZED_LOGPROB.name] > b[LENGTH_NORMALIZED_LOGPROB.name]


def test_length_penalty_alpha_zero_recovers_the_raw_total() -> None:
    assert length_penalty(7, 0.0) == pytest.approx(1.0)


def test_mean_and_max_entropy_differ_on_a_spiky_span() -> None:
    tokens = (
        _token([1.0]),  # entropy 0
        _token([0.5, 0.5]),  # entropy ln2
    )
    out = compute_family_a(_greedy(tokens), SPEC)
    assert out[MEAN_TOP5_ENTROPY.name] == pytest.approx(math.log(2) / 2)
    assert out[MAX_TOP5_ENTROPY.name] == pytest.approx(math.log(2))


# ---------------------------------------------------- empty answer and inf


def test_empty_answer_span_is_nan_everywhere_not_zero() -> None:
    """0.0 is a legitimate value for several of these, so it cannot mean missing.

    A mean logprob of 0.0 is a perfectly confident answer and a margin of 0.0 is
    a dead heat. Returning 0.0 for an empty span would place those rows at
    opposite extremes of two different rankings, both wrong.
    """
    out = compute_family_a(_greedy(()), SPEC)
    assert set(out) == {s.name for s in FAMILY_A_SIGNALS}
    for name, value in out.items():
        assert math.isnan(value), f"{name} returned {value!r} for an empty span"


def test_perplexity_clamps_instead_of_overflowing_to_inf() -> None:
    """One inf poisons every bootstrap resample that draws the row.

    A float16 underflow upstream can hand us a logprob near -800. exp(800) is
    inf, and inf propagates silently through the mean, the correlation matrix and
    the logistic fit. The clamp keeps it finite and last-ranked.
    """
    pathological = _greedy((TokenLogprob("a", -800.0, {"a": -800.0}),))
    value = compute_family_a(pathological, SPEC)[PERPLEXITY.name]
    assert math.isfinite(value)
    assert value == pytest.approx(math.exp(SPEC.perplexity_clamp))


def test_perplexity_below_the_clamp_is_untouched() -> None:
    tokens = (TokenLogprob("a", -2.0, {"a": -2.0}),)
    assert compute_family_a(_greedy(tokens), SPEC)[PERPLEXITY.name] == pytest.approx(math.exp(2.0))


# ------------------------------------------------------------ orientation


def test_every_family_a_signal_is_registered_in_family_a() -> None:
    names = set(signal_names(FAMILY_A))
    assert names == {s.name for s in FAMILY_A_SIGNALS}


def test_orientation_is_declared_for_every_family_a_signal() -> None:
    for spec in FAMILY_A_SIGNALS:
        assert spec.orientation in (ORIENT_RISK, ORIENT_CONFIDENCE)


def test_oriented_values_all_increase_with_wrongness() -> None:
    """The property that matters: a confident answer scores lower than a shaky one.

    An inverted signal reports AUROC 1-x, which reads as a strong result upside
    down and is invisible in the table. This asserts the sign of every family-A
    signal at once against a pair of generations where confidence is unambiguous.
    """
    confident = _greedy(
        (
            TokenLogprob("Paris", -0.01, {"Paris": -0.01, "Lyon": -6.0}),
            TokenLogprob("!", -0.02, {"!": -0.02, "?": -5.0}),
        )
    )
    shaky = _greedy(
        (
            TokenLogprob("Ouag", -2.30, {"Ouag": -2.30, "Bam": -2.31}),
            TokenLogprob("adou", -2.40, {"adou": -2.40, "ako": -2.41}),
        )
    )
    lo = orient_all(compute_family_a(confident, SPEC))
    hi = orient_all(compute_family_a(shaky, SPEC))
    for name in lo:
        assert lo[name] < hi[name], f"{name} is oriented backwards: {lo[name]} !< {hi[name]}"


def test_orientation_flips_confidence_and_leaves_risk_alone() -> None:
    assert get_spec(MEAN_LOGPROB.name).oriented(-1.5) == pytest.approx(1.5)
    assert get_spec(PERPLEXITY.name).oriented(4.0) == pytest.approx(4.0)
    assert get_spec(FIRST_TOKEN_MARGIN.name).oriented(0.8) == pytest.approx(-0.8)


def test_orientation_passes_nan_through_unchanged() -> None:
    for spec in FAMILY_A_SIGNALS:
        assert math.isnan(spec.oriented(float("nan")))


def test_orient_all_rejects_an_unregistered_signal() -> None:
    with pytest.raises(KeyError):
        orient_all({"a_signal_that_does_not_exist": 1.0})
