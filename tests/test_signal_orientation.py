"""One orientation test per signal (defect D2).

Run #1 published a verbalized-confidence AUROC of 0.386, well below chance. A
signal that scores that far under 0.5 is usually not a weak signal, it is a
sign-flipped one, and the way to find out is to test the flip rather than argue
about it in prose. `signals.base` already centralizes the flip in
`SignalSpec.oriented`, so what was missing was a test that the declared
orientation is the true one for every signal in the registry.

Two layers here, and the second is the one that catches a real bug.

The first layer is mechanical: `oriented()` must negate a confidence signal and
pass a risk signal through. That only checks the plumbing.

The second layer is behavioural. For each signal a synthetic dataset is built in
which the underlying quantity is deliberately correlated with correctness in the
direction the signal's own semantics imply — a high mean logprob means the model
was confident and therefore more often right, a high semantic entropy means the
samples disagreed and therefore more often wrong. The oriented value must then
produce an AUROC above 0.5 against the INCORRECT positive class. If a spec
declares the wrong orientation the oriented AUROC lands below 0.5 and the test
fails, which is precisely the run #1 failure mode.

The registry is enumerated rather than hard-coded, so a signal added later
without an orientation test fails `test_every_registered_signal_is_covered`
instead of silently going unchecked.
"""

from __future__ import annotations

import numpy as np
import pytest

from unc_bench.analysis.metrics import auroc
from unc_bench.signals.base import (
    ORIENT_CONFIDENCE,
    ORIENT_RISK,
    get_spec,
    signal_names,
)
from unc_bench.stages import score_signals  # noqa: F401  (registers every signal)

#: For each signal: does the RAW value go up when the answer is more likely to
#: be CORRECT? This table is written from each signal's meaning, independently of
#: what `SignalSpec.orientation` claims, so that the two can be cross-checked.
#: That independence is the point — copying the spec would make the test vacuous.
RAW_RISES_WITH_CORRECTNESS: dict[str, bool] = {
    # Family A: token logprobs. Higher logprob = more confident = more often right.
    "a_mean_logprob": True,
    "a_min_logprob": True,
    "a_total_logprob": True,
    "a_length_normalized_logprob": True,
    "a_first_token_logprob": True,
    "a_first_token_margin": True,
    # Perplexity and entropy are inverse-confidence quantities.
    "a_perplexity": False,
    "a_mean_top5_entropy": False,
    "a_max_top5_entropy": False,
    # Family B: self-consistency. Agreement means right, disagreement means wrong.
    "b_mean_pairwise_f1": True,
    "b_disagreement_rate": False,
    "b_distinct_count": False,
    "b_distinct_fraction": False,
    "b_semantic_entropy": False,
    "b_semantic_entropy_normalized": False,
    "b_mean_pairwise_f1_samples_only": True,
    "b_disagreement_rate_samples_only": False,
    "b_distinct_count_samples_only": False,
    "b_distinct_fraction_samples_only": False,
    "b_semantic_entropy_samples_only": False,
    "b_semantic_entropy_normalized_samples_only": False,
    # Family C: self-verification. All three are confidence statements.
    "c_p_true_plain": True,
    "c_p_true_with_samples": True,
    "c_verbal_confidence": True,
    # Trivial baselines. Long answers and long questions are hedges and hard
    # questions respectively, so both are risk-shaped; random is neither, and is
    # excluded from the behavioural layer below.
    "t_answer_length": False,
    "t_question_length": False,
    "t_random": False,
}


def test_every_registered_signal_is_covered() -> None:
    """No signal may skip its orientation test by being added quietly."""
    missing = sorted(set(signal_names()) - set(RAW_RISES_WITH_CORRECTNESS))
    assert not missing, f"signals with no orientation expectation: {missing}"
    extra = sorted(set(RAW_RISES_WITH_CORRECTNESS) - set(signal_names()))
    assert not extra, f"orientation expectations for unregistered signals: {extra}"


@pytest.mark.parametrize("name", sorted(RAW_RISES_WITH_CORRECTNESS))
def test_declared_orientation_matches_semantics(name: str) -> None:
    """The spec's orientation must agree with what the signal actually means.

    A confidence-oriented signal is one whose raw value rises with correctness.
    This is the check that would have caught a sign-flipped verbalized
    confidence at definition time.
    """
    spec = get_spec(name)
    rises_with_correctness = RAW_RISES_WITH_CORRECTNESS[name]
    expected = ORIENT_CONFIDENCE if rises_with_correctness else ORIENT_RISK
    assert spec.orientation == expected, (
        f"{name} raw value {'rises' if rises_with_correctness else 'falls'} with "
        f"correctness, so orientation should be {expected!r}, not {spec.orientation!r}"
    )


@pytest.mark.parametrize("name", sorted(RAW_RISES_WITH_CORRECTNESS))
def test_oriented_value_is_negated_only_for_confidence(name: str) -> None:
    """The plumbing layer: `oriented()` flips confidence and passes risk through."""
    spec = get_spec(name)
    probe = 0.37
    if spec.orientation == ORIENT_CONFIDENCE:
        assert spec.oriented(probe) == pytest.approx(-probe)
    else:
        assert spec.oriented(probe) == pytest.approx(probe)


@pytest.mark.parametrize("name", sorted(n for n in RAW_RISES_WITH_CORRECTNESS if n != "t_random"))
def test_oriented_signal_scores_above_chance_on_synthetic_data(name: str) -> None:
    """The behavioural layer, and the one that matters.

    Build rows where the raw quantity moves with correctness in the direction the
    signal's semantics demand, then require the ORIENTED column to beat chance
    against the INCORRECT positive class. Getting this wrong is what produced run
    #1's 0.386.

    `t_random` is excluded because it has no true direction; its own test below
    asserts it sits at chance instead.
    """
    spec = get_spec(name)
    rng = np.random.default_rng(20250830)
    n = 400
    incorrect = np.zeros(n, dtype=bool)
    incorrect[: n // 2] = True

    # Raw values: separated means, in the direction the signal's meaning implies.
    high_when_correct = RAW_RISES_WITH_CORRECTNESS[name]
    correct_mean, incorrect_mean = (2.0, 0.0) if high_when_correct else (0.0, 2.0)
    raw = np.where(
        incorrect,
        rng.normal(incorrect_mean, 0.7, n),
        rng.normal(correct_mean, 0.7, n),
    )

    oriented = np.array([spec.oriented(float(v)) for v in raw], dtype=np.float64)
    score = auroc(oriented, incorrect)
    assert score > 0.5, (
        f"{name}: oriented AUROC {score:.3f} is at or below chance on data built to "
        f"favour it, which means the declared orientation {spec.orientation!r} is "
        f"backwards"
    )
    # On this much separation a correctly oriented signal should be far above
    # chance, not marginally above it.
    assert score > 0.85, f"{name}: oriented AUROC {score:.3f} unexpectedly weak"


def test_random_baseline_sits_at_chance() -> None:
    """`t_random` must be uninformative regardless of orientation.

    This is the signal that exposed run #1: with 7 positives it scored 0.746. The
    assertion here is on synthetic balanced data, so it checks the signal itself
    rather than the run, and the run-level version of the same check is a
    validity gate in the analysis stage.
    """
    spec = get_spec("t_random")
    rng = np.random.default_rng(7)
    n = 4000
    incorrect = np.zeros(n, dtype=bool)
    incorrect[: n // 2] = True
    raw = rng.uniform(0.0, 1.0, n)
    oriented = np.array([spec.oriented(float(v)) for v in raw], dtype=np.float64)
    score = auroc(oriented, incorrect)
    assert abs(score - 0.5) < 0.05, f"t_random scored {score:.3f}, which is not chance"
