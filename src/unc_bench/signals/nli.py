"""Bidirectional-entailment clustering for semantic entropy.

Two things in here are load-bearing and easy to get silently wrong.

The entailment label index. Several MNLI checkpoints on the Hub publish
`{0: entailment, 1: neutral, 2: contradiction}` and several publish the exact
reverse. Hard-coding index 0 works on half of them and inverts every clustering
decision on the other half, with no error and a plausible-looking output: the
most self-consistent items get the highest semantic entropy. So the index is
resolved by NAME from `config.id2label` and the resolution raises if no label
matches.

Exact duplicates bypass the model. An MNLI checkpoint does not reliably entail a
string against itself — "Paris" vs "Paris" can land at 0.4 entailment on the base
checkpoint, below any sensible threshold. If that noise reached the clustering,
five identical samples would sometimes split into five clusters and the model's
most confident answers would be handed the highest semantic entropy. Normalized
string equality is a stronger signal than any NLI score and is checked first.
"""

from __future__ import annotations

from typing import Protocol

from unc_bench.config import NLISpec


class EntailmentModel(Protocol):
    """Directional entailment probability for premise -> hypothesis pairs.

    Batched, because the real implementation runs a transformer and one forward
    pass per pair on 2 CPU cores is the difference between 0.09 s and 0.5 s.
    """

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis) pair, in order."""
        ...


def resolve_entailment_index(id2label: dict[int, str], expected: str = "entailment") -> int:
    """Find the entailment logit index by name.

    Raises rather than guessing. A wrong index here is undetectable downstream:
    the numbers stay in range, the clusters stay plausible, and every conclusion
    inverts.
    """
    wanted = expected.strip().casefold()
    for index, label in id2label.items():
        if str(label).strip().casefold() == wanted:
            return int(index)
    raise ValueError(
        f"no label named {expected!r} in id2label={id2label!r}; "
        "refusing to guess the entailment index"
    )


class DebertaEntailmentModel:
    """`MoritzLaurer/DeBERTa-v3-base-mnli` on CPU.

    Downgraded from `-large`, which does not co-reside with the generator in
    2 GB (docs/DECISIONS.md D9). Requires `sentencepiece`: the DeBERTa-v3
    tokenizer is sentencepiece-backed and `AutoTokenizer` raises a bare import
    error without it.
    """

    def __init__(self, spec: NLISpec) -> None:
        # Deferred import: CI installs neither torch nor transformers.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._spec = spec
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(spec.model_name)
        self._model.eval()
        id2label = {int(k): str(v) for k, v in self._model.config.id2label.items()}
        self.id2label = id2label
        self.entailment_index = resolve_entailment_index(id2label, spec.entailment_label)

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        torch = self._torch
        out: list[float] = []
        for start in range(0, len(pairs), self._spec.batch_size):
            chunk = pairs[start : start + self._spec.batch_size]
            encoded = self._tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._spec.max_length,
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self.entailment_index]
            out.extend(float(p) for p in probs.tolist())
        return out


class ScriptedEntailmentModel:
    """Test double whose entailment scores are written down, not learned.

    Semantic-entropy clustering is defined by an asymmetry: A may entail B while
    B does not entail A, and the pair must then stay in separate clusters. You
    cannot construct a guaranteed-asymmetric pair against real weights. "Paris"
    entails "Paris, France" at ~0.99 with essentially nothing in the reverse
    direction, but that is an empirical fact about one checkpoint, not something a
    test can assert without pinning the weights. Scripting the scores makes the
    clustering logic testable in CI with no model download at all.

    Unlisted pairs score `default`.
    """

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        *,
        default: float = 0.0,
    ) -> None:
        self.scores = dict(scores)
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def entailment_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.extend(pairs)
        return [self.scores.get(pair, self.default) for pair in pairs]
