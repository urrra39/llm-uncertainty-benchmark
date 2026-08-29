"""Three-stage labeling, plus abstention as its own category.

Stage 0 is abstention. `UNKNOWN` is not an error and not a correct answer. It is
a refusal, and it gets its own label. Counting refusals as errors inflates every
AUROC in the study, because a refusal is trivially predictable from any of these
signals — an empty or boilerplate answer has a distinctive logprob profile and
five identical refusals have zero semantic entropy. A benchmark that scores that
as "signal predicts error" is measuring its own tautology. So abstentions are
reported separately and the main analysis runs twice, with and without them.

Stage 1 is normalized exact match against the gold aliases. Free, and it settles
the majority of items.

Stage 2 sends the survivors to a judge at temperature 0 with a rubric that
permits three verdicts: correct, incorrect, ambiguous. The third one exists
because PopQA aliases are genuinely incomplete and forcing a binary verdict on
an item the rubric cannot settle manufactures label noise. Ambiguous items are
dropped from the analysis and their count is reported.

Stage 3 re-labels a subsample with a second, independent judge and computes
Cohen's kappa. Below the configured threshold the README says plainly that the
headline numbers are not trustworthy.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from unc_bench.config import Config, JudgeSpec
from unc_bench.normalize import clean_model_answer, exact_match, is_abstention, normalize_answer
from unc_bench.types import (
    LABEL_ABSTAIN,
    LABEL_AMBIGUOUS,
    LABEL_CORRECT,
    LABEL_INCORRECT,
    SOURCE_ABSTENTION,
    SOURCE_EXACT_MATCH,
    Label,
    Question,
)

JUDGE_PROMPT = """You are grading one short factual answer against a gold answer list.

Question: {question}
Gold answers (any one is acceptable): {gold}
Model answer: {answer}

Rules:
- Reply CORRECT if the model answer names the same entity, quantity or date as
  any gold answer, allowing for spelling, abbreviation, word order, and extra
  qualifiers that do not change the referent.
- Reply INCORRECT if it names a different entity, is empty, refuses, or answers
  a different question.
- Reply AMBIGUOUS only if the gold list looks incomplete and you cannot tell
  whether the model answer is an acceptable alternative.

Reply with exactly one word: CORRECT, INCORRECT, or AMBIGUOUS."""


class TextJudge(Protocol):
    """A judge only needs to return text. No logprobs required.

    This is why the labeling stage works on the available API gateway while
    signal family A does not: the gateway returns no `logprobs` object for any
    model it allows, but it returns text just fine.
    """

    @property
    def model_name(self) -> str: ...

    def complete_text(
        self, prompt: str, *, seed: int = 0, max_new_tokens: int | None = None
    ) -> str: ...


# Anchored alternatives over the whole normalized reply, not a substring search.
# "incorrect" CONTAINS "correct", so `if "correct" in reply` labels every wrong
# answer correct. That single bug would invert the majority of the label set and
# leave every AUROC hovering near 0.5 with no visible cause.
_VERDICT_RE = re.compile(r"^(correct|incorrect|ambiguous)$")

# The verdict as the FIRST word of the reply, with the rest of the reply
# ignored. Anchored at the start and requiring a word boundary, so "incorrect"
# can never be read as "correct": the alternation is ordered longest-first and
# the boundary stops a prefix match mid-word.
_VERDICT_LEAD_RE = re.compile(r"^(incorrect|correct|ambiguous)\b")


def parse_verdict(reply: str) -> str | None:
    """Map a judge reply to a label, or None if it does not parse.

    The reply is normalized and matched whole first. A judge that returns
    "CORRECT." or " correct " is fine.

    claude-haiku-4-5 ignores the "exactly one word" instruction on about three
    quarters of items and answers "INCORRECT\\n\\nThe model answer ... does not
    match", which the whole-string match rejected. That dropped the second
    judge to 15 usable verdicts out of 60 and made Cohen's kappa a statement
    about the 15 items where it happened to comply. So a leading verdict is
    accepted as the verdict. Anything that does not START with one of the three
    words is still a parse failure and is still reported as one, rather than
    being mined for a keyword anywhere in the reply.
    """
    text = normalize_answer(reply or "")
    match: re.Match[str] | None = _VERDICT_RE.match(text)
    if match is None:
        match = _VERDICT_LEAD_RE.match(text)
    if match is None:
        return None
    verdict = match.group(1)
    return {
        "correct": LABEL_CORRECT,
        "incorrect": LABEL_INCORRECT,
        "ambiguous": LABEL_AMBIGUOUS,
    }[verdict]


def label_by_exact_match(question: Question, raw_answer: str, abstain_token: str) -> Label | None:
    """Stage 0 and stage 1. Returns None when the item needs a judge.

    Abstention is checked first and short-circuits. An abstention that happened
    to string-match a gold alias would otherwise be scored correct, and while
    that is vanishingly unlikely for "UNKNOWN", the ordering states the intent.
    """
    answer = clean_model_answer(raw_answer, abstain_token=abstain_token)
    if is_abstention(answer, abstain_token):
        return Label(qid=question.qid, value=LABEL_ABSTAIN, source=SOURCE_ABSTENTION)
    if exact_match(answer, question.gold_answers):
        return Label(qid=question.qid, value=LABEL_CORRECT, source=SOURCE_EXACT_MATCH)
    # An exact-match MISS is not an error yet. "Bram Stoker" against a gold list
    # of ["Stoker"] is a miss and a correct answer, which is what stage 2 is for.
    return None


def render_judge_prompt(question: Question, answer: str) -> str:
    return JUDGE_PROMPT.format(
        question=question.question,
        gold=" | ".join(question.gold_answers),
        answer=answer if answer.strip() else "(empty)",
    )


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """One judge verdict plus enough context to audit it."""

    qid: str
    value: str | None
    raw: str
    parse_failed: bool


def judge_item(judge: TextJudge, question: Question, answer: str, *, seed: int) -> JudgeOutcome:
    """One judge call at temperature 0.

    A parse failure is recorded, not defaulted. Silently mapping an unparseable
    verdict to "incorrect" would push the base rate up by whatever the failure
    rate happens to be and there would be no trace of it in the results.
    """
    prompt = render_judge_prompt(question, answer)
    try:
        raw = judge.complete_text(prompt, seed=seed, max_new_tokens=8)
    except Exception as exc:
        # Broad on purpose: a judge outage must not abort a run that has already
        # spent hours on generation. The failure is recorded and reported.
        return JudgeOutcome(qid=question.qid, value=None, raw=f"error: {exc}", parse_failed=True)
    value = parse_verdict(raw)
    return JudgeOutcome(qid=question.qid, value=value, raw=raw, parse_failed=value is None)


# ------------------------------------------------------------- fuzzy fallback

# Judge gateway unusable: fall back to exact match plus this rule, and mark the
# whole label set heuristic in the README. The kappa claim is dropped rather
# than faked, because there is no second judge to disagree with.

# Largest token-count difference the containment rule will accept. Two covers
# the real cases ("Bram Stoker" vs "Stoker", "the United States of America" vs
# "United States") without letting a sentence swallow a one-word alias.
MAX_LENGTH_GAP = 2


def fuzzy_correct(answer: str, gold_answers: Sequence[str]) -> bool:
    """Containment either way on normalized token sequences.

    Catches the two dominant exact-match misses: a model answer that adds a
    qualifier ("Bram Stoker" vs "Stoker") and one that drops one ("Clinton" vs
    "Bill Clinton"). Deliberately does NOT do token-overlap scoring, which would
    score "Paris, Texas" against "Paris, France" as a partial match; containment
    of whole token sequences is strict enough to keep those apart.

    Containment is capped by the LENGTH GAP, not by which side is longer. My
    first attempt capped `len(gold) <= len(pred) + 2`, which is trivially true
    whenever the gold alias is short and therefore constrained nothing: gold
    "Rome" was contained in "Rome was not built in a day you know" and scored
    correct. The cap has to bound the difference in both directions, so a match
    is only allowed when the two token sequences are within `MAX_LENGTH_GAP` of
    each other.
    """
    predicted = normalize_answer(answer)
    if not predicted:
        return False
    pred_tokens = predicted.split()
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        gold_tokens = gold_norm.split()
        if gold_tokens == pred_tokens:
            return True
        if abs(len(pred_tokens) - len(gold_tokens)) > MAX_LENGTH_GAP:
            continue
        if _contains(pred_tokens, gold_tokens) or _contains(gold_tokens, pred_tokens):
            return True
    return False


def _contains(haystack: list[str], needle: list[str]) -> bool:
    """Contiguous-subsequence test on token lists."""
    if not needle or len(needle) > len(haystack):
        return False
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


def label_heuristic(question: Question, raw_answer: str, abstain_token: str) -> Label:
    """Full label without a judge. Used only when the gateway is unusable."""
    early = label_by_exact_match(question, raw_answer, abstain_token)
    if early is not None:
        return early
    answer = clean_model_answer(raw_answer, abstain_token=abstain_token)
    value = LABEL_CORRECT if fuzzy_correct(answer, question.gold_answers) else LABEL_INCORRECT
    return Label(qid=question.qid, value=value, source="heuristic_fuzzy")


# ----------------------------------------------------------- Cohen's kappa


@dataclass(frozen=True, slots=True)
class KappaResult:
    """Cohen's kappa with the counts needed to interpret it."""

    kappa: float
    n: int
    observed_agreement: float
    expected_agreement: float
    categories: tuple[str, ...]
    trustworthy: bool


def cohens_kappa(a: Sequence[str], b: Sequence[str], *, threshold: float = 0.7) -> KappaResult:
    """Cohen's kappa between two label sequences.

    The degenerate case is handled explicitly. If both judges assign every item
    the same single category, observed agreement is 1.0 and expected agreement is
    also 1.0, so kappa is 0/0. That is not "no agreement" — it is unanimity with
    no variance to measure — and reporting 0.0 would read as two judges
    disagreeing completely. NaN with a note is the honest output, and it is
    marked untrustworthy so no one quotes it.
    """
    if len(a) != len(b):
        raise ValueError(f"kappa needs paired labels: got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("kappa over an empty sample is undefined")

    categories = tuple(sorted(set(a) | set(b)))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = 0.0
    for category in categories:
        p_a = sum(1 for x in a if x == category) / n
        p_b = sum(1 for y in b if y == category) / n
        expected += p_a * p_b

    if expected >= 1.0:
        return KappaResult(
            kappa=float("nan"),
            n=n,
            observed_agreement=observed,
            expected_agreement=expected,
            categories=categories,
            trustworthy=False,
        )
    kappa = (observed - expected) / (1.0 - expected)
    return KappaResult(
        kappa=kappa,
        n=n,
        observed_agreement=observed,
        expected_agreement=expected,
        categories=categories,
        trustworthy=kappa >= threshold,
    )


def cross_validation_sample(qids: Sequence[str], spec: JudgeSpec) -> list[str]:
    """Deterministic subsample of qids for the second judge.

    Sorted before sampling so the choice does not depend on row order, matching
    the dataset builders' convention.
    """
    import numpy as np

    ordered = sorted(qids)
    n = min(spec.cross_validation_n, len(ordered))
    rng = np.random.default_rng(spec.secondary_seed)
    picked = rng.choice(len(ordered), size=n, replace=False)
    return [ordered[int(i)] for i in sorted(picked)]


def build_judge(spec: Config, *, secondary: bool = False) -> TextJudge:
    """Construct a judge client from the config."""
    from unc_bench.client import OpenAICompatibleClient

    model_spec = spec.judges.secondary if secondary else spec.judges.primary
    return OpenAICompatibleClient(model_spec)
