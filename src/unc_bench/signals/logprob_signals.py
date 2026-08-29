"""Signal family A: everything readable off one greedy generation's logprobs.

Cost is 1x. These are the signals you get for free if you are already generating
an answer, which is why they are the baseline every more expensive family has to
beat.

Three constraints hold throughout.

The answer span only. `Generation.answer_token_logprobs` never contains prompt
tokens, so no aggregate here can average over instruction boilerplate. That is
enforced upstream in the client, not re-checked here.

Empty answer returns NaN, never 0.0. An empty generation has no mean logprob,
and 0.0 is a legitimate value for a mean logprob (a perfectly confident token),
for the top1-top2 margin, and for entropy. Substituting 0.0 would put those rows
at the extreme confident end of the ranking rather than marking them missing.

Orientation is declared once, in `signals.base`, and applied only through
`SignalSpec.oriented`. Nothing in this module negates a value.
"""

from __future__ import annotations

import math

from unc_bench.config import SignalsSpec
from unc_bench.signals.base import (
    FAMILY_A,
    ORIENT_CONFIDENCE,
    ORIENT_RISK,
    SignalSpec,
    register,
)
from unc_bench.types import Generation, TokenLogprob

MEAN_LOGPROB = register(
    SignalSpec(
        name="a_mean_logprob",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="mean per-token logprob over the answer span",
    )
)
MIN_LOGPROB = register(
    SignalSpec(
        name="a_min_logprob",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="least-likely answer token; the weakest link in the span",
    )
)
TOTAL_LOGPROB = register(
    SignalSpec(
        name="a_total_logprob",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="sum of answer-token logprobs, i.e. log P(answer)",
    )
)
LENGTH_NORMALIZED_LOGPROB = register(
    SignalSpec(
        name="a_length_normalized_logprob",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="total logprob under the Wu et al. length penalty",
    )
)
PERPLEXITY = register(
    SignalSpec(
        name="a_perplexity",
        family=FAMILY_A,
        orientation=ORIENT_RISK,
        description="exp(-mean logprob) over the answer span",
    )
)
FIRST_TOKEN_LOGPROB = register(
    SignalSpec(
        name="a_first_token_logprob",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="logprob of the first answer token",
    )
)
MEAN_TOP5_ENTROPY = register(
    SignalSpec(
        name="a_mean_top5_entropy",
        family=FAMILY_A,
        orientation=ORIENT_RISK,
        description="mean truncated predictive entropy over answer positions",
    )
)
MAX_TOP5_ENTROPY = register(
    SignalSpec(
        name="a_max_top5_entropy",
        family=FAMILY_A,
        orientation=ORIENT_RISK,
        description="most uncertain single answer position",
    )
)
FIRST_TOKEN_MARGIN = register(
    SignalSpec(
        name="a_first_token_margin",
        family=FAMILY_A,
        orientation=ORIENT_CONFIDENCE,
        description="p(top1) - p(top2) at the first answer position",
    )
)

FAMILY_A_SIGNALS: tuple[SignalSpec, ...] = (
    MEAN_LOGPROB,
    MIN_LOGPROB,
    TOTAL_LOGPROB,
    LENGTH_NORMALIZED_LOGPROB,
    PERPLEXITY,
    FIRST_TOKEN_LOGPROB,
    MEAN_TOP5_ENTROPY,
    MAX_TOP5_ENTROPY,
    FIRST_TOKEN_MARGIN,
)


def _renormalized_top_probs(token: TokenLogprob) -> list[float]:
    """Descending probabilities over the renormalized top-k at one position.

    The top-k logprobs do not sum to 1, so they are renormalized. Everything
    derived from them is therefore a truncated-support approximation and is
    labelled as such wherever it is reported.
    """
    if not token.top_logprobs:
        return []
    probs = [math.exp(lp) for lp in token.top_logprobs.values()]
    total = sum(probs)
    if total <= 0.0:
        return []
    return sorted((p / total for p in probs), reverse=True)


def top1_top2_margin(token: TokenLogprob) -> float:
    """p(top1) - p(top2) in probability space, not logprob space.

    Probability space is the defensible choice: a logprob difference of 2 nats
    means something completely different at p=0.9 than at p=0.002, and the
    quantity is supposed to express "how close was the runner-up". A single
    observed alternative means the runner-up is unobserved; the margin is then
    reported against 0.0, which is the correct reading of a top-k that only
    contained one entry.
    """
    probs = _renormalized_top_probs(token)
    if not probs:
        return float("nan")
    if len(probs) == 1:
        return probs[0]
    return probs[0] - probs[1]


def truncated_entropy(token: TokenLogprob) -> float:
    """Shannon entropy in nats over the renormalized top-k at one position."""
    return token.entropy_over_top_k()


def length_penalty(n_tokens: int, alpha: float) -> float:
    """Wu et al. 2016 length penalty, lp(n) = ((5 + n) / 6) ** alpha.

    Worth being explicit about why this is not just `total / n`: total logprob
    divided by token count IS the mean logprob, so a signal defined that way
    would be a duplicate column with a different name and would show up in the
    correlation heatmap at exactly 1.00. The Wu penalty is sublinear in n, so it
    penalizes length less aggressively than the mean does and the two signals
    genuinely rank rows differently. alpha=0 recovers the raw total, alpha=1 is
    close to the mean.
    """
    if n_tokens <= 0:
        return float("nan")
    return float(((5.0 + n_tokens) / 6.0) ** alpha)


def compute_family_a(gen: Generation, spec: SignalsSpec) -> dict[str, float]:
    """Every family-A signal for one greedy generation, in raw units.

    Raw means each value is in its natural direction. Orientation is applied by
    the caller through `signals.base.orient_all`.
    """
    tokens = gen.answer_token_logprobs
    nan = float("nan")
    if not tokens:
        # No answer span: every one of these is undefined. NaN, not 0.0.
        return {s.name: nan for s in FAMILY_A_SIGNALS}

    logprobs = [t.logprob for t in tokens]
    n = len(logprobs)
    total = math.fsum(logprobs)
    mean = total / n
    entropies = [truncated_entropy(t) for t in tokens]

    return {
        MEAN_LOGPROB.name: mean,
        MIN_LOGPROB.name: min(logprobs),
        TOTAL_LOGPROB.name: total,
        LENGTH_NORMALIZED_LOGPROB.name: total / length_penalty(n, spec.length_penalty_alpha),
        PERPLEXITY.name: _perplexity(mean, spec.perplexity_clamp),
        FIRST_TOKEN_LOGPROB.name: logprobs[0],
        MEAN_TOP5_ENTROPY.name: math.fsum(entropies) / n,
        MAX_TOP5_ENTROPY.name: max(entropies),
        FIRST_TOKEN_MARGIN.name: top1_top2_margin(tokens[0]),
    }


def _perplexity(mean_logprob: float, clamp: float) -> float:
    """exp(-mean logprob), clamped before the exponential.

    The clamp is not cosmetic. A single pathological token (a logprob of -800,
    which a float16 underflow can produce) sends `exp(-mean)` to `inf`. One inf
    in a signal column poisons every bootstrap resample that draws that row,
    every correlation involving that column, and the logistic regression fit, and
    it does so without raising. Clamping produces a very large but finite value,
    which ranks the row last — the intended reading — and keeps the arithmetic
    downstream well-defined.
    """
    if math.isnan(mean_logprob):
        return float("nan")
    exponent = min(-mean_logprob, clamp)
    return math.exp(exponent)
