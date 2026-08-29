"""Family C and the trivial baselines.

The two assertions worth the most in here are the ones that raise: a multi-token
True spelling, and True and False colliding on one id. Neither produces a wrong
number that looks wrong. The collision in particular pins P(True) at exactly 0.5
for every item in the study, which reads as a clean "this signal is
uninformative" finding and is in fact a tokenizer bug.
"""

from __future__ import annotations

import math

import pytest

from unc_bench.config import SignalsSpec
from unc_bench.signals.base import FAMILY_C, FAMILY_T, orient_all, signal_names
from unc_bench.signals.trivial import (
    ANSWER_LENGTH,
    FAMILY_T_SIGNALS,
    QUESTION_LENGTH,
    RANDOM,
    compute_family_t,
)
from unc_bench.signals.verification import (
    FALSE_SPELLINGS,
    FAMILY_C_SIGNALS,
    P_TRUE_PLAIN,
    P_TRUE_WITH_SAMPLES,
    TRUE_SPELLINGS,
    VERBAL_CONFIDENCE,
    compute_family_c,
    parse_verbal_confidence,
    renormalized_p_true,
    resolve_true_false_tokens,
    score_p_true,
)
from unc_bench.types import VerificationResult

SPEC = SignalsSpec()


class FakeScorer:
    """Tokenizer and single-token-logprob stub.

    `vocab` maps a spelling to a token id; anything absent is treated as
    multi-token. `logprobs` maps a token id to its logprob at the next position.
    """

    def __init__(self, vocab: dict[str, int], logprobs: dict[int, float] | None = None) -> None:
        self.vocab = dict(vocab)
        self.logprobs = dict(logprobs or {})
        self.scored: list[tuple[str, int]] = []

    def token_ids_for(self, text: str) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        return [1, 2]  # stands in for a multi-token spelling

    def logprob_of_token_id(self, prompt: str, token_id: int) -> float:
        self.scored.append((prompt, token_id))
        return self.logprobs.get(token_id, -20.0)


QWEN_LIKE = {" True": 3007, " False": 3557}


# ------------------------------------------------------ token resolution


def test_leading_space_spelling_is_preferred() -> None:
    """A chat template makes " True" the natural continuation, so try it first.

    Both spellings exist as single tokens on Qwen2.5 (" True"=3007, "True"=2514)
    and they carry different amounts of probability mass. Picking the bare one
    would measure the wrong token.
    """
    scorer = FakeScorer({" True": 3007, "True": 2514, " False": 3557, "False": 4049})
    tokens = resolve_true_false_tokens(scorer)
    assert (tokens.true_spelling, tokens.true_id) == (" True", 3007)
    assert (tokens.false_spelling, tokens.false_id) == (" False", 3557)


def test_falls_back_through_the_spelling_list() -> None:
    scorer = FakeScorer({"true": 830, "FALSE": 30351})
    tokens = resolve_true_false_tokens(scorer)
    assert tokens.true_spelling == "true"
    assert tokens.false_spelling == "FALSE"


def test_raises_when_no_true_spelling_is_single_token() -> None:
    """A multi-token spelling makes P(True) the logprob of a prefix."""
    scorer = FakeScorer({" False": 3557})
    with pytest.raises(RuntimeError, match="no single-token spelling of True"):
        resolve_true_false_tokens(scorer)


def test_raises_when_no_false_spelling_is_single_token() -> None:
    scorer = FakeScorer({" True": 3007})
    with pytest.raises(RuntimeError, match="no single-token spelling of False"):
        resolve_true_false_tokens(scorer)


def test_raises_when_true_and_false_share_a_token_id() -> None:
    """The expensive silent failure: P(True) pinned at 0.5 for every item.

    p/(p+p) is 0.5 regardless of the logprob, so the signal comes out perfectly
    flat, the AUROC comes out at exactly 0.50, and the result reads as
    "self-verification carries no information" rather than as a broken lookup.
    """
    scorer = FakeScorer({" True": 999, " False": 999})
    with pytest.raises(RuntimeError, match="same token id"):
        resolve_true_false_tokens(scorer)


def test_spelling_lists_start_with_the_leading_space_variant() -> None:
    assert TRUE_SPELLINGS[0] == " True"
    assert FALSE_SPELLINGS[0] == " False"


# -------------------------------------------------------- renormalization


def test_equal_logprobs_give_exactly_one_half() -> None:
    assert renormalized_p_true(-3.0, -3.0) == pytest.approx(0.5)


def test_renormalization_matches_hand_computed_value() -> None:
    # p_true = e^-1, p_false = e^-2; e^-1/(e^-1+e^-2) = 1/(1+e^-1) = 0.7310585786
    assert renormalized_p_true(-1.0, -2.0) == pytest.approx(0.7310585786300049, abs=1e-12)


def test_renormalization_is_not_the_raw_exponential() -> None:
    """The whole point of family C's implementation, in one assertion.

    A 0.5B model puts most of its mass on "Yes" or a newline, so both logprobs
    here are around -12 and exp(-12) is 6e-6. Read raw, that says the model is
    almost certain the answer is wrong. Renormalized over the pair it says 0.88,
    which is what the model was actually asked.
    """
    true_lp, false_lp = -12.0, -14.0
    assert math.exp(true_lp) < 1e-5
    assert renormalized_p_true(true_lp, false_lp) == pytest.approx(0.8807970779778823, abs=1e-12)


def test_renormalization_survives_two_very_small_probabilities() -> None:
    """Log-space shift, not exp(a)/(exp(a)+exp(b)).

    At -700 each, both exp() terms underflow to 0.0 and the naive form is 0/0.
    """
    assert renormalized_p_true(-700.0, -700.0) == pytest.approx(0.5)
    assert renormalized_p_true(-700.0, -701.0) == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_renormalization_propagates_nan() -> None:
    assert math.isnan(renormalized_p_true(float("nan"), -1.0))


def test_score_p_true_queries_both_resolved_ids() -> None:
    scorer = FakeScorer(QWEN_LIKE, {3007: -0.5, 3557: -1.5})
    tokens = resolve_true_false_tokens(scorer)
    scorer.scored.clear()
    result = score_p_true("prompt", scorer, tokens)
    assert [tid for _, tid in scorer.scored] == [3007, 3557]
    assert result.p_true == pytest.approx(renormalized_p_true(-0.5, -1.5))
    assert result.parse_failed is False


# ---------------------------------------------------- verbal confidence


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("90", 0.90),
        (" 90 ", 0.90),
        ("90%", 0.90),
        ("0", 0.0),
        ("100", 1.0),
        ("7", 0.07),
    ],
)
def test_verbal_confidence_parses_a_bare_integer(reply: str, expected: float) -> None:
    assert parse_verbal_confidence(reply, SPEC) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "about 90",
        "I'd say 85, maybe",
        "90.5",
        "ninety",
        "101",
        "-5",
        "90 percent confident",
        "Confidence: 90",
    ],
)
def test_verbal_confidence_returns_none_on_a_parse_failure(reply: str) -> None:
    """None, not 0.5.

    A default would be indistinguishable from a genuine "exactly undecided"
    reply and would pile every unparseable row onto the midpoint of the
    distribution. The failure RATE is reported instead.
    """
    assert parse_verbal_confidence(reply, SPEC) is None


def test_verbal_confidence_rejects_prose_containing_a_number() -> None:
    """Strict is the point. A model that cannot follow "a single integer" is
    telling you about its instruction-following, and digging the number out of
    the prose with a loose regex hides that."""
    assert parse_verbal_confidence("The answer is 42 and I am 90% sure", SPEC) is None


# --------------------------------------------------------- family C rows


def test_family_c_row_carries_both_variants_and_the_verbal_value() -> None:
    out = compute_family_c(
        VerificationResult(p_true=0.8, true_token=" True", false_token=" False"),
        VerificationResult(p_true=0.6, true_token=" True", false_token=" False"),
        0.9,
    )
    assert out[P_TRUE_PLAIN.name] == pytest.approx(0.8)
    assert out[P_TRUE_WITH_SAMPLES.name] == pytest.approx(0.6)
    assert out[VERBAL_CONFIDENCE.name] == pytest.approx(0.9)


def test_family_c_missing_values_are_nan() -> None:
    out = compute_family_c(
        None, VerificationResult(p_true=None, true_token=None, false_token=None), None
    )
    assert set(out) == {s.name for s in FAMILY_C_SIGNALS}
    for name, value in out.items():
        assert math.isnan(value), f"{name} returned {value!r}"


def test_every_family_c_signal_is_registered_in_family_c() -> None:
    assert set(signal_names(FAMILY_C)) == {s.name for s in FAMILY_C_SIGNALS}


def test_oriented_family_c_values_increase_with_wrongness() -> None:
    confident = orient_all(
        compute_family_c(
            VerificationResult(p_true=0.95, true_token=" True", false_token=" False"),
            VerificationResult(p_true=0.93, true_token=" True", false_token=" False"),
            0.95,
        )
    )
    doubtful = orient_all(
        compute_family_c(
            VerificationResult(p_true=0.10, true_token=" True", false_token=" False"),
            VerificationResult(p_true=0.12, true_token=" True", false_token=" False"),
            0.20,
        )
    )
    for name in confident:
        assert confident[name] < doubtful[name], f"{name} is oriented backwards"


# ------------------------------------------------------------- family T


def test_lengths_are_counted_in_normalized_tokens() -> None:
    out = compute_family_t("Who wrote Dracula?", "Bram Stoker", qid="q1", seed=7)
    # "who wrote dracula" after article stripping and punctuation removal.
    assert out[QUESTION_LENGTH.name] == pytest.approx(3.0)
    assert out[ANSWER_LENGTH.name] == pytest.approx(2.0)


def test_random_baseline_is_reproducible_from_qid_and_seed() -> None:
    """Same qid and seed, same value, in any process.

    The seed goes through SHA-256 rather than the builtin `hash()`, which is
    salted per process. With `hash()` this test passes inside one run and the
    column silently differs between runs.
    """
    a = compute_family_t("q", "a", qid="popqa-1", seed=7)[RANDOM.name]
    b = compute_family_t("q", "a", qid="popqa-1", seed=7)[RANDOM.name]
    assert a == b


def test_random_baseline_does_not_depend_on_row_order() -> None:
    """Different qids get independent draws, and no shared stream couples them.

    With a shared generator, inserting one question upstream would shift every
    later row's noise and the baseline would not be reproducible from the config.
    """
    first = compute_family_t("q", "a", qid="popqa-1", seed=7)[RANDOM.name]
    _ = compute_family_t("q", "a", qid="popqa-99", seed=7)
    again = compute_family_t("q", "a", qid="popqa-1", seed=7)[RANDOM.name]
    assert first == again


def test_random_baseline_changes_with_the_seed() -> None:
    a = compute_family_t("q", "a", qid="popqa-1", seed=7)[RANDOM.name]
    b = compute_family_t("q", "a", qid="popqa-1", seed=8)[RANDOM.name]
    assert a != b


def test_random_baseline_is_on_the_unit_interval() -> None:
    values = [compute_family_t("q", "a", qid=f"q{i}", seed=1)[RANDOM.name] for i in range(50)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert len(set(values)) == 50


def test_empty_answer_has_zero_length_not_nan() -> None:
    """Length is genuinely 0 for an empty answer, unlike a mean logprob.

    This is the one signal where 0.0 is the correct value for an empty
    generation rather than a missing-value stand-in.
    """
    out = compute_family_t("Who wrote Dracula?", "", qid="q1", seed=7)
    assert out[ANSWER_LENGTH.name] == 0.0


def test_every_family_t_signal_is_registered_in_family_t() -> None:
    assert set(signal_names(FAMILY_T)) == {s.name for s in FAMILY_T_SIGNALS}


class BatchedScorer(FakeScorer):
    """Scorer that exposes the one-forward-pass lookup."""

    def __init__(self, vocab: dict[str, int], logprobs: dict[int, float]) -> None:
        super().__init__(vocab, logprobs)
        self.batch_calls = 0

    def logprobs_of_token_ids(self, prompt: str, token_ids: list[int]) -> list[float]:
        del prompt  # stub; the arg exists to match the protocol
        self.batch_calls += 1
        return [self.logprobs.get(i, -20.0) for i in token_ids]


def test_score_p_true_uses_the_batched_lookup_when_available() -> None:
    """Both logprobs are at the same position of the same prompt.

    Two separate calls run an identical forward pass twice, which measured at
    4.55 s against 2.3 s on this machine. Over two variants and 250 questions
    that is 19 minutes for no information.
    """
    scorer = BatchedScorer(QWEN_LIKE, {3007: -0.5, 3557: -1.5})
    tokens = resolve_true_false_tokens(scorer)
    scorer.scored.clear()
    result = score_p_true("prompt", scorer, tokens)
    assert scorer.batch_calls == 1
    assert scorer.scored == []  # the per-token path was not used
    assert result.p_true == pytest.approx(renormalized_p_true(-0.5, -1.5))


def test_score_p_true_falls_back_to_the_per_token_path() -> None:
    scorer = FakeScorer(QWEN_LIKE, {3007: -0.5, 3557: -1.5})
    tokens = resolve_true_false_tokens(scorer)
    scorer.scored.clear()
    result = score_p_true("prompt", scorer, tokens)
    assert len(scorer.scored) == 2
    assert result.p_true == pytest.approx(renormalized_p_true(-0.5, -1.5))
