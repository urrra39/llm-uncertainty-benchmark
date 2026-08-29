"""Typed records that cross stage boundaries.

Each pipeline stage reads and writes parquet. These dataclasses define the
schema. The important design point is `TokenLogprob`: the raw per-token
distribution is persisted, not just the aggregates. That is what makes it
possible to add a new signal in family A later without regenerating anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    """One generated token, its logprob, and the top-k alternatives at that step.

    `top_logprobs` maps token text to logprob and includes the chosen token.
    """

    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)

    def entropy_over_top_k(self) -> float:
        """Shannon entropy in nats over the renormalized top-k distribution.

        The top-k probabilities do not sum to 1, so they are renormalized. This
        is a truncated-support approximation to the full predictive entropy and
        is labelled as such wherever it is reported.
        """
        if not self.top_logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in self.top_logprobs.values()]
        total = sum(probs)
        if total <= 0.0:
            return 0.0
        out = 0.0
        for p in probs:
            q = p / total
            if q > 0.0:
                out -= q * math.log(q)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "logprob": self.logprob,
            "top_logprobs": dict(self.top_logprobs),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TokenLogprob:
        return cls(
            token=str(raw["token"]),
            logprob=float(raw["logprob"]),
            top_logprobs={str(k): float(v) for k, v in (raw.get("top_logprobs") or {}).items()},
        )


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark item, after dataset building."""

    qid: str
    dataset: str
    question: str
    gold_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.gold_answers:
            raise ValueError(f"{self.qid}: no gold answers")
        if not self.question.strip():
            raise ValueError(f"{self.qid}: empty question")


@dataclass(frozen=True, slots=True)
class Generation:
    """A single model response.

    `answer_token_logprobs` covers the answer span ONLY. Prompt tokens are never
    included, which is the whole point: averaging logprobs over prompt or
    boilerplate tokens is one of the failure modes this project tests for.
    """

    text: str
    answer_token_logprobs: tuple[TokenLogprob, ...]
    is_greedy: bool
    seed: int
    temperature: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        if self.is_greedy and self.temperature != 0.0:
            raise ValueError("a generation marked greedy must have temperature 0.0")

    @property
    def n_answer_tokens(self) -> int:
        return len(self.answer_token_logprobs)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Output of a self-verification call.

    `p_true` is read off the logprobs of the single True/False token, not parsed
    from text, so it is a real probability rather than a coarse label.
    """

    p_true: float | None
    true_token: str | None
    false_token: str | None
    raw_text: str = ""
    parse_failed: bool = False


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    """Everything generated for one question, before signal scoring."""

    question: Question
    greedy: Generation
    samples: tuple[Generation, ...]
    verify_plain: VerificationResult | None = None
    verify_with_samples: VerificationResult | None = None
    verbal_confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.greedy.is_greedy:
            raise ValueError(f"{self.question.qid}: greedy slot holds a sampled generation")
        for s in self.samples:
            if s.is_greedy:
                raise ValueError(f"{self.question.qid}: sample slot holds a greedy generation")


# Label vocabulary. `ambiguous` items are dropped from the main analysis and
# their count is reported.
LabelValue = str
LABEL_CORRECT: LabelValue = "correct"
LABEL_INCORRECT: LabelValue = "incorrect"
LABEL_AMBIGUOUS: LabelValue = "ambiguous"
LABEL_ABSTAIN: LabelValue = "abstain"
ALL_LABELS: tuple[LabelValue, ...] = (
    LABEL_CORRECT,
    LABEL_INCORRECT,
    LABEL_AMBIGUOUS,
    LABEL_ABSTAIN,
)

# How a label was reached, so the exact-match-only view can be reconstructed.
SOURCE_EXACT_MATCH = "exact_match"
SOURCE_JUDGE = "judge"
SOURCE_ABSTENTION = "abstention"


@dataclass(frozen=True, slots=True)
class Label:
    qid: str
    value: LabelValue
    source: str
    judge_raw: str = ""

    def __post_init__(self) -> None:
        if self.value not in ALL_LABELS:
            raise ValueError(f"{self.qid}: unknown label {self.value!r}")
