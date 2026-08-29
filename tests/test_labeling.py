"""Labeling, verdict parsing, and Cohen's kappa.

The single most important test in this file is
`test_incorrect_is_not_parsed_as_correct`. "incorrect" contains "correct" as a
substring, so the obvious implementation labels every wrong answer correct. That
inverts most of the label set and leaves every AUROC sitting near 0.5 with
nothing in the output to explain why.
"""

from __future__ import annotations

import math

import pytest

from unc_bench.config import JudgeSpec, ModelSpec
from unc_bench.labeling import (
    JUDGE_PROMPT,
    cohens_kappa,
    cross_validation_sample,
    fuzzy_correct,
    judge_item,
    label_by_exact_match,
    label_heuristic,
    parse_verdict,
    render_judge_prompt,
)
from unc_bench.types import (
    LABEL_ABSTAIN,
    LABEL_AMBIGUOUS,
    LABEL_CORRECT,
    LABEL_INCORRECT,
    SOURCE_ABSTENTION,
    SOURCE_EXACT_MATCH,
    Question,
)

DRACULA = Question(
    qid="q1",
    dataset="triviaqa",
    question="Who wrote Dracula?",
    gold_answers=("Bram Stoker", "Stoker"),
)


class ScriptedJudge:
    """Returns a canned reply, or raises to simulate a gateway outage."""

    def __init__(self, reply: str | None = None, *, raises: bool = False) -> None:
        self.reply = reply or ""
        self.raises = raises
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted-judge"

    def complete_text(
        self, prompt: str, *, seed: int = 0, max_new_tokens: int | None = None
    ) -> str:
        del seed, max_new_tokens
        self.prompts.append(prompt)
        if self.raises:
            raise RuntimeError("gateway 503")
        return self.reply


# ------------------------------------------------------- verdict parsing


def test_incorrect_is_not_parsed_as_correct() -> None:
    """The substring trap. `"correct" in "incorrect"` is True.

    A substring check here labels every wrong answer correct, which inverts the
    majority of the label set and drives every AUROC toward 0.5. Nothing in the
    output would point at the cause.
    """
    assert parse_verdict("INCORRECT") == LABEL_INCORRECT
    assert parse_verdict("incorrect") == LABEL_INCORRECT
    assert parse_verdict("CORRECT") == LABEL_CORRECT


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("CORRECT", LABEL_CORRECT),
        ("correct", LABEL_CORRECT),
        (" Correct ", LABEL_CORRECT),
        ("CORRECT.", LABEL_CORRECT),
        ("INCORRECT", LABEL_INCORRECT),
        ("AMBIGUOUS", LABEL_AMBIGUOUS),
        ("ambiguous!", LABEL_AMBIGUOUS),
    ],
)
def test_verdict_parses_the_whole_normalized_reply(reply: str, expected: str) -> None:
    assert parse_verdict(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "The answer is correct because Bram Stoker wrote it",
        "yes",
        "1",
        "I cannot grade this",
    ],
)
def test_verdict_returns_none_on_a_parse_failure(reply: str) -> None:
    """Leading-verdict match, not keyword mining.

    A reply that mentions a verdict somewhere in the middle of prose is still a
    parse failure. Only a reply that OPENS with one of the three words counts,
    because that is the shape claude-haiku-4-5 actually produces when it ignores
    the one-word instruction: the verdict first, then an unsolicited
    justification. Mining a keyword from anywhere in the prose would let "the
    answer is correct because" through, and it does not get through.
    """
    assert parse_verdict(reply) is None


# --------------------------------------------------- exact-match stages


def test_abstention_gets_its_own_label_not_incorrect() -> None:
    """Counting refusals as errors inflates every AUROC.

    A refusal is trivially predictable from any of these signals: the logprob
    profile is distinctive and five identical refusals have zero semantic
    entropy. Scoring that as "signal predicts error" measures a tautology.
    """
    label = label_by_exact_match(DRACULA, "UNKNOWN", "UNKNOWN")
    assert label is not None
    assert label.value == LABEL_ABSTAIN
    assert label.source == SOURCE_ABSTENTION


def test_abstention_matching_is_normalized() -> None:
    for raw in ("unknown", "UNKNOWN.", "  Unknown  ", "Answer: UNKNOWN"):
        label = label_by_exact_match(DRACULA, raw, "UNKNOWN")
        assert label is not None and label.value == LABEL_ABSTAIN, raw


def test_a_longer_answer_containing_unknown_is_not_an_abstention() -> None:
    """ "The Unknown Soldier" is a real answer, not a refusal."""
    assert label_by_exact_match(DRACULA, "The Unknown Soldier", "UNKNOWN") is None


def test_exact_match_settles_a_hit_without_a_judge() -> None:
    label = label_by_exact_match(DRACULA, "Bram Stoker", "UNKNOWN")
    assert label is not None
    assert label.value == LABEL_CORRECT
    assert label.source == SOURCE_EXACT_MATCH


def test_exact_match_matches_any_alias() -> None:
    label = label_by_exact_match(DRACULA, "stoker", "UNKNOWN")
    assert label is not None and label.value == LABEL_CORRECT


def test_an_exact_match_miss_is_not_yet_an_error() -> None:
    """The reason stage 2 exists.

    "Bram Stoker, the Irish novelist" misses every alias by exact match and is
    a perfectly correct answer. Labeling it incorrect here would put real
    accuracy well below the truth and add noise to every signal at once.
    """
    assert label_by_exact_match(DRACULA, "Bram Stoker, the Irish novelist", "UNKNOWN") is None


def test_answer_prefix_is_stripped_before_matching() -> None:
    label = label_by_exact_match(DRACULA, "Answer: Bram Stoker", "UNKNOWN")
    assert label is not None and label.value == LABEL_CORRECT


# ------------------------------------------------------------ judge call


def test_judge_prompt_carries_the_question_gold_list_and_answer() -> None:
    prompt = render_judge_prompt(DRACULA, "Bram Stoker")
    assert "Who wrote Dracula?" in prompt
    assert "Bram Stoker | Stoker" in prompt
    assert "AMBIGUOUS" in prompt


def test_empty_answer_is_rendered_visibly_in_the_prompt() -> None:
    """An empty slot after "Model answer:" reads as a truncated prompt.

    The judge would then be grading a question with no answer attached and its
    verdict would be arbitrary.
    """
    assert "(empty)" in render_judge_prompt(DRACULA, "   ")


def test_judge_rubric_permits_exactly_three_verdicts() -> None:
    for verdict in ("CORRECT", "INCORRECT", "AMBIGUOUS"):
        assert verdict in JUDGE_PROMPT


def test_judge_item_records_a_parse_failure_rather_than_defaulting() -> None:
    """Defaulting to "incorrect" would move the base rate by the failure rate
    and leave no trace of it in the results."""
    outcome = judge_item(ScriptedJudge("I think it's fine"), DRACULA, "Bram Stoker", seed=0)
    assert outcome.value is None
    assert outcome.parse_failed is True
    assert outcome.raw == "I think it's fine"


def test_judge_item_survives_a_gateway_outage() -> None:
    """One 503 must not abort a run that has already spent hours on generation."""
    outcome = judge_item(ScriptedJudge(raises=True), DRACULA, "Bram Stoker", seed=0)
    assert outcome.value is None
    assert outcome.parse_failed is True
    assert "gateway 503" in outcome.raw


def test_judge_item_returns_the_parsed_verdict() -> None:
    outcome = judge_item(ScriptedJudge("AMBIGUOUS"), DRACULA, "Stoker family", seed=0)
    assert outcome.value == LABEL_AMBIGUOUS
    assert outcome.parse_failed is False


# ------------------------------------------------------- fuzzy fallback


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Bram Stoker", ("Stoker",), True),
        ("Stoker", ("Bram Stoker",), True),
        ("Bill Clinton", ("Clinton",), True),
        ("Rome", ("Rome",), True),
        ("Paris, Texas", ("Paris, France",), False),
        ("Jules Verne", ("Bram Stoker",), False),
        ("", ("Stoker",), False),
    ],
)
def test_fuzzy_rule_catches_qualifiers_without_merging_distinct_entities(
    answer: str, gold: tuple[str, ...], expected: bool
) -> None:
    assert fuzzy_correct(answer, gold) is expected


def test_fuzzy_rule_will_not_let_a_long_answer_swallow_a_short_alias() -> None:
    """Without the length cap, gold "Rome" is contained in this and scores
    correct, which would make the heuristic fallback useless."""
    assert fuzzy_correct("Rome was not built in a single day you know", ("Rome",)) is False


def test_heuristic_labeling_still_routes_abstentions_correctly() -> None:
    label = label_heuristic(DRACULA, "UNKNOWN", "UNKNOWN")
    assert label.value == LABEL_ABSTAIN


def test_heuristic_labeling_marks_its_source() -> None:
    """The README must be able to say which labels were heuristic."""
    label = label_heuristic(DRACULA, "Bram Stoker, the novelist", "UNKNOWN")
    assert label.source == "heuristic_fuzzy"


# --------------------------------------------------------------- kappa


def test_kappa_of_perfect_agreement_with_variance_is_one() -> None:
    a = [LABEL_CORRECT, LABEL_INCORRECT, LABEL_CORRECT, LABEL_INCORRECT]
    assert cohens_kappa(a, list(a)).kappa == pytest.approx(1.0)


def test_kappa_matches_a_hand_computed_two_by_two_table() -> None:
    # 20 items: 8 both-correct, 7 both-incorrect, 3 and 2 off-diagonal.
    # observed = 15/20 = 0.75
    # judge A correct 11/20, judge B correct 10/20
    # expected = 0.55*0.50 + 0.45*0.50 = 0.5
    # kappa = (0.75 - 0.5) / 0.5 = 0.5
    a = [LABEL_CORRECT] * 11 + [LABEL_INCORRECT] * 9
    b = [LABEL_CORRECT] * 8 + [LABEL_INCORRECT] * 3 + [LABEL_CORRECT] * 2 + [LABEL_INCORRECT] * 7
    result = cohens_kappa(a, b)
    assert result.observed_agreement == pytest.approx(0.75)
    assert result.expected_agreement == pytest.approx(0.5)
    assert result.kappa == pytest.approx(0.5)


def test_kappa_of_unanimity_with_no_variance_is_nan_not_zero() -> None:
    """Two judges who both say "correct" every time have not disagreed.

    Observed agreement is 1.0 and so is expected, giving 0/0. Reporting 0.0
    would read as total disagreement, which is the opposite of what happened.
    NaN plus `trustworthy=False` stops anyone quoting the number.
    """
    a = [LABEL_CORRECT] * 30
    result = cohens_kappa(a, list(a))
    assert math.isnan(result.kappa)
    assert result.trustworthy is False
    assert result.observed_agreement == pytest.approx(1.0)


def test_kappa_flags_itself_untrustworthy_below_the_threshold() -> None:
    a = [LABEL_CORRECT] * 11 + [LABEL_INCORRECT] * 9
    b = [LABEL_CORRECT] * 8 + [LABEL_INCORRECT] * 3 + [LABEL_CORRECT] * 2 + [LABEL_INCORRECT] * 7
    assert cohens_kappa(a, b, threshold=0.7).trustworthy is False
    assert cohens_kappa(a, b, threshold=0.4).trustworthy is True


def test_kappa_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="paired"):
        cohens_kappa([LABEL_CORRECT], [LABEL_CORRECT, LABEL_INCORRECT])


def test_kappa_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty"):
        cohens_kappa([], [])


def test_kappa_handles_three_categories() -> None:
    a = [LABEL_CORRECT, LABEL_INCORRECT, LABEL_AMBIGUOUS, LABEL_CORRECT]
    result = cohens_kappa(a, list(a))
    assert result.categories == tuple(sorted({LABEL_CORRECT, LABEL_INCORRECT, LABEL_AMBIGUOUS}))
    assert result.kappa == pytest.approx(1.0)


# --------------------------------------------------- cross-validation sample


def _judge_spec(n: int) -> JudgeSpec:
    return JudgeSpec(
        primary=ModelSpec(backend="openai_compatible", name="gpt-5-mini"),
        secondary=ModelSpec(backend="openai_compatible", name="claude-haiku-4-5"),
        cross_validation_n=n,
    )


def test_cross_validation_sample_is_deterministic_and_order_independent() -> None:
    qids = [f"q{i}" for i in range(200)]
    spec = _judge_spec(100)
    first = cross_validation_sample(qids, spec)
    shuffled = cross_validation_sample(list(reversed(qids)), spec)
    assert first == shuffled
    assert len(first) == 100
    assert len(set(first)) == 100


def test_cross_validation_sample_caps_at_the_available_rows() -> None:
    picked = cross_validation_sample([f"q{i}" for i in range(40)], _judge_spec(100))
    assert len(picked) == 40
