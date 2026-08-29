"""Answer normalization and exact match.

This is the SQuAD/TriviaQA normalization: lowercase, strip articles, strip
punctuation, collapse whitespace. It is deliberately conservative. Anything
looser starts scoring "Paris, France" as equal to "Paris, Texas".

Everything downstream leans on this. Exact match produces stage-1 labels,
self-consistency agreement counts distinct normalized answers, and NLI
clustering compares normalized strings, so a bug here moves every number in the
results table at once.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterable, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

# Strip every Unicode punctuation codepoint, not just the ASCII set.
# string.punctuation misses U+2019 RIGHT SINGLE QUOTATION MARK, which is what
# models actually emit in names like O(U+2019)Brien, and NFKC does not fold it to an
# ASCII apostrophe. Built once at import; the table is ~800 entries.
_PUNCT_TABLE = {
    codepoint: None
    for codepoint in range(sys.maxunicode + 1)
    if unicodedata.category(chr(codepoint)).startswith("P")
}


def normalize_answer(text: str) -> str:
    """Canonical form used for every string comparison in the project.

    Unicode is NFKC-folded first so that a curly apostrophe, a non-breaking
    space or a full-width digit does not read as a different answer.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.casefold()
    out = out.translate(_PUNCT_TABLE)
    # Articles are stripped after punctuation, so "the, Beatles" also works.
    # Note the consequence: a bare token "a" is an article and disappears. That
    # is correct for prose answers and irrelevant for real gold spans, but it
    # does mean single-letter test strings are a poor choice of fixture.
    out = _ARTICLES.sub(" ", out)
    out = _WHITESPACE.sub(" ", out)
    return out.strip()


def tokenize(text: str) -> list[str]:
    """Whitespace tokens of the normalized form."""
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def is_abstention(text: str, abstain_token: str = "UNKNOWN") -> bool:
    """True when the model declined to answer.

    Matched on the normalized form, so "unknown", "UNKNOWN." and " Unknown "
    all count. A longer answer that merely contains the word does not: "unknown
    soldier" is a real answer and must not be silently dropped as an
    abstention.
    """
    normalized = normalize_answer(text)
    return normalized == normalize_answer(abstain_token)


def exact_match(prediction: str, gold_answers: Iterable[str]) -> bool:
    """Normalized exact match against any gold alias."""
    predicted = normalize_answer(prediction)
    if not predicted:
        return False
    return any(predicted == normalize_answer(g) for g in gold_answers)


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between two answers, on normalized tokens.

    Multiset semantics: a repeated token in both strings counts twice. Two
    empty strings score 1.0 (identical), one empty scores 0.0.
    """
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    counts_b: dict[str, int] = {}
    for token in tokens_b:
        counts_b[token] = counts_b.get(token, 0) + 1
    overlap = 0
    for token in tokens_a:
        remaining = counts_b.get(token, 0)
        if remaining > 0:
            overlap += 1
            counts_b[token] = remaining - 1

    if overlap == 0:
        return 0.0
    precision = overlap / len(tokens_a)
    recall = overlap / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def mean_pairwise_token_f1(answers: Sequence[str]) -> float:
    """Mean token F1 over all unordered pairs.

    A single answer has no pairs; 1.0 is returned because a set of one is
    trivially self-consistent, and the alternative (0.0) would read as maximal
    disagreement.
    """
    if len(answers) < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            total += token_f1(answers[i], answers[j])
            pairs += 1
    return total / pairs


def strip_answer_prefix(text: str) -> str:
    """Remove a leading 'A:' or 'Answer:' the model sometimes echoes.

    Applied before normalization so that the echoed label does not become an
    answer token and inflate the length baseline.
    """
    cleaned = text.strip()
    for prefix in ("answer:", "a:", "the answer is"):
        if cleaned.casefold().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    return cleaned


def first_line(text: str) -> str:
    """First non-empty line. Short-form answers are one line by construction;
    anything after it is the model ignoring the instruction."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def clean_model_answer(text: str, *, abstain_token: str = "UNKNOWN") -> str:
    """Full extraction path from raw generation text to a comparable answer."""
    candidate = strip_answer_prefix(first_line(text))
    if is_abstention(candidate, abstain_token):
        return abstain_token
    return candidate


def count_distinct(answers: Iterable[str]) -> int:
    """Number of distinct normalized answers."""
    return len({normalize_answer(a) for a in answers})
