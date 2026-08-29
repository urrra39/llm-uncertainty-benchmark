"""Signal family C: ask the model whether its own answer is right.

Two P(True) variants after Kadavath et al. 2022, "Language Models (Mostly) Know
What They Know": one showing only the proposed answer, one also showing the five
sampled answers as context. Cost is about 1.1x, since a verification is a single
forward pass rather than a generation.

The renormalization is the part that matters. `exp(logprob(" True"))` is not
P(True). On a 0.5B model most of the next-token mass at that position goes to
"Yes", to a newline, or to a restatement of the question, so the raw probability
of " True" sits around 0.3 even when the model is confidently affirming. Read as
a probability that says "70% chance this answer is wrong", which is not what the
model said at all. Renormalizing over the {True, False} pair asks the question
that was actually posed: given that you must pick one of these two words, which?

    P(True) = p_true / (p_true + p_false)

The verbalized-confidence variant is kept even though it is expected to be
degenerate at this model scale. A constant 90 is a finding about small models,
and dropping the signal would hide it.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from unc_bench.config import SignalsSpec
from unc_bench.signals.base import (
    FAMILY_C,
    ORIENT_CONFIDENCE,
    SignalSpec,
    register,
)
from unc_bench.types import VerificationResult

P_TRUE_PLAIN = register(
    SignalSpec(
        name="c_p_true_plain",
        family=FAMILY_C,
        orientation=ORIENT_CONFIDENCE,
        description="renormalized P(True) shown only the proposed answer",
    )
)
P_TRUE_WITH_SAMPLES = register(
    SignalSpec(
        name="c_p_true_with_samples",
        family=FAMILY_C,
        orientation=ORIENT_CONFIDENCE,
        description="renormalized P(True) also shown the N sampled answers",
    )
)
VERBAL_CONFIDENCE = register(
    SignalSpec(
        name="c_verbal_confidence",
        family=FAMILY_C,
        orientation=ORIENT_CONFIDENCE,
        description="self-reported 0-100 confidence, parsed strictly",
    )
)

FAMILY_C_SIGNALS: tuple[SignalSpec, ...] = (
    P_TRUE_PLAIN,
    P_TRUE_WITH_SAMPLES,
    VERBAL_CONFIDENCE,
)

# Spellings tried in order, leading space first. A chat template ends with a
# newline-free assistant prefix, so " True" is the natural continuation and is
# what the model actually puts mass on; the bare "True" is a different token id
# and usually a worse one. Casing variants come last because they are rarely
# single tokens outside byte-level BPE.
TRUE_SPELLINGS: tuple[str, ...] = (" True", "True", " true", "true", " TRUE", "TRUE")
FALSE_SPELLINGS: tuple[str, ...] = (" False", "False", " false", "false", " FALSE", "FALSE")


class TokenScorer(Protocol):
    """What P(True) needs: a token-id lookup and a single-token logprob."""

    def token_ids_for(self, text: str) -> list[int]: ...

    def logprob_of_token_id(self, prompt: str, token_id: int) -> float: ...


class BatchTokenScorer(Protocol):
    """A scorer that can return several token logprobs from one forward pass."""

    def logprobs_of_token_ids(self, prompt: str, token_ids: list[int]) -> list[float]: ...


class TrueFalseTokens:
    """Resolved single-token ids for True and False, with the spellings used.

    Resolution happens once per run, not per question, because it is a property
    of the tokenizer.
    """

    def __init__(
        self,
        *,
        true_id: int,
        false_id: int,
        true_spelling: str,
        false_spelling: str,
    ) -> None:
        self.true_id = true_id
        self.false_id = false_id
        self.true_spelling = true_spelling
        self.false_spelling = false_spelling

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"TrueFalseTokens({self.true_spelling!r}={self.true_id}, "
            f"{self.false_spelling!r}={self.false_id})"
        )


def resolve_true_false_tokens(scorer: TokenScorer) -> TrueFalseTokens:
    """Find single-token spellings of True and False, or raise.

    Both failure modes raise loudly, and both need to. A multi-token spelling
    means the "logprob of the True token" is the logprob of a prefix and the
    quantity is not what it claims to be. A shared id between the two spellings
    is worse: the renormalization becomes p/(p+p) and P(True) is pinned at
    exactly 0.5 for every item in the study. That reads as a perfectly
    uninformative signal — AUROC 0.50, no error, no warning — rather than as a
    broken one, and it is the kind of result you would write a paragraph about
    before noticing it was a bug.

    On Qwen2.5 this resolves to " True" = 3007 and " False" = 3557, verified
    against the tokenizer rather than assumed.
    """
    true_id: int | None = None
    true_spelling = ""
    for spelling in TRUE_SPELLINGS:
        ids = scorer.token_ids_for(spelling)
        if len(ids) == 1:
            true_id, true_spelling = ids[0], spelling
            break
    false_id: int | None = None
    false_spelling = ""
    for spelling in FALSE_SPELLINGS:
        ids = scorer.token_ids_for(spelling)
        if len(ids) == 1:
            false_id, false_spelling = ids[0], spelling
            break

    if true_id is None:
        raise RuntimeError(
            f"no single-token spelling of True among {TRUE_SPELLINGS}; "
            "P(True) would be the logprob of a prefix, not of the word"
        )
    if false_id is None:
        raise RuntimeError(
            f"no single-token spelling of False among {FALSE_SPELLINGS}; "
            "P(True) would be the logprob of a prefix, not of the word"
        )
    if true_id == false_id:
        raise RuntimeError(
            f"True ({true_spelling!r}) and False ({false_spelling!r}) resolved to the "
            f"same token id {true_id}; P(True) would be pinned at 0.5 for every item"
        )
    return TrueFalseTokens(
        true_id=true_id,
        false_id=false_id,
        true_spelling=true_spelling,
        false_spelling=false_spelling,
    )


def renormalized_p_true(true_logprob: float, false_logprob: float) -> float:
    """P(True) over the {True, False} pair.

    Computed in log space via a shifted exponential rather than as
    exp(a)/(exp(a)+exp(b)). Both logprobs can be around -12 on a small model,
    where exp() of each is ~6e-6 and the ratio of two tiny floats loses
    precision. Subtracting the larger first keeps one term at exactly 1.0.
    """
    if math.isnan(true_logprob) or math.isnan(false_logprob):
        return float("nan")
    top = max(true_logprob, false_logprob)
    p_true = math.exp(true_logprob - top)
    p_false = math.exp(false_logprob - top)
    total = p_true + p_false
    if total <= 0.0:  # pragma: no cover - one term is 1.0 by construction
        return float("nan")
    return p_true / total


def score_p_true(prompt: str, scorer: TokenScorer, tokens: TrueFalseTokens) -> VerificationResult:
    """One verification forward pass, read off the True/False logprobs.

    Uses the batched lookup when the scorer offers one. Both logprobs come from
    the same position of the same prompt, so two separate calls would run an
    identical forward pass twice; on this machine that is 2.3 s wasted per
    variant, per question.
    """
    batched = getattr(scorer, "logprobs_of_token_ids", None)
    if callable(batched):
        true_lp, false_lp = batched(prompt, [tokens.true_id, tokens.false_id])
    else:
        true_lp = scorer.logprob_of_token_id(prompt, tokens.true_id)
        false_lp = scorer.logprob_of_token_id(prompt, tokens.false_id)
    return VerificationResult(
        p_true=renormalized_p_true(true_lp, false_lp),
        true_token=tokens.true_spelling,
        false_token=tokens.false_spelling,
        raw_text="",
        parse_failed=False,
    )


# A bare integer, optionally with a percent sign or surrounding whitespace.
# Anchored: a reply that says anything else is a parse failure, not a number to
# be dug out of prose.
_CONFIDENCE_RE = re.compile(r"^\s*(\d{1,3})\s*%?\s*$")


def parse_verbal_confidence(text: str, spec: SignalsSpec) -> float | None:
    """Parse a 0-100 confidence reply, or return None.

    None, never 0.5. A default would be indistinguishable from a genuine "I am
    exactly undecided" reply, and it would silently convert a high parse-failure
    rate into a large spike of rows at the midpoint. The failure rate is reported
    instead, which is the honest version of the same information.

    Strict on purpose. "About 90%" and "I'd say 85, maybe" are failures. A model
    that cannot follow "reply with a single integer" is telling you something
    about its instruction-following, and quietly regexing a number out of the
    prose hides that.
    """
    match = _CONFIDENCE_RE.match(text or "")
    if match is None:
        return None
    value = int(match.group(1))
    if not 0 <= value <= spec.verbal_confidence_max:
        return None
    return value / spec.verbal_confidence_max


def compute_family_c(
    verify_plain: VerificationResult | None,
    verify_with_samples: VerificationResult | None,
    verbal_confidence: float | None,
) -> dict[str, float]:
    """Every family-C signal, in raw units. Missing values are NaN."""
    nan = float("nan")

    def _p(result: VerificationResult | None) -> float:
        if result is None or result.p_true is None:
            return nan
        return float(result.p_true)

    return {
        P_TRUE_PLAIN.name: _p(verify_plain),
        P_TRUE_WITH_SAMPLES.name: _p(verify_with_samples),
        VERBAL_CONFIDENCE.name: nan if verbal_confidence is None else float(verbal_confidence),
    }
