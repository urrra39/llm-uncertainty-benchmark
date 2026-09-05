"""Validity gates, enforced as assertions rather than described in prose (D15).

Run #1's failure was not that a number came out badly. It was that the run
produced a full 21-signal ranking on 7 positive rows, and nothing in the
pipeline objected. The random baseline scored 0.746 AUROC — a value only
reachable when the sample is too small for AUROC to mean anything — and that
number sat in a results table next to the real signals as though it were a
finding.

So the gates live in code, they run as part of the analysis stage, and their
outcome is written into results.json where the README generator has to read it.
Three of them:

1. The random baseline's CI must contain 0.50. This is the direct test of
   whether the measurement apparatus works at all. `t_random` is pure noise by
   construction, so if its interval excludes chance then the interval is too
   narrow, the sample is too small, or the labels are broken — and in every one
   of those cases no other AUROC in the table can be trusted either.
2. At least 30 positives and 30 negatives. Below that the bootstrap interval is
   wide enough to admit almost any ordering of the signals.
3. Abstention rate below 10%. Refusals are trivially predictable and inflate
   every AUROC, so a high rate means the table is partly measuring refusal
   detection.

A failed gate does not raise. It records `passed: false` with the observed
value, and the README must then print VALIDITY FAILED and publish no ranking.
Raising would be the wrong behaviour: the failure is the result, and it has to
be publishable rather than fatal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: The random baseline signal. Its whole purpose is to fail informatively.
RANDOM_SIGNAL = "t_random"

#: Minimum rows per class before a ranking is considered powered.
MIN_PER_CLASS = 30

#: Ceiling on the abstention rate, as a fraction of scored rows.
MAX_ABSTENTION_RATE = 0.10

#: Floor on the fraction of validation rows carrying a human label. Labels are
#: the ground the AUROC table stands on; a machine-only label set is unmeasured
#: correctness no matter how high the judge-versus-judge kappa reads. 0.80 is a
#: judgement call and is stated as one.
MIN_HUMAN_LABEL_COVERAGE = 0.80


@dataclass(frozen=True, slots=True)
class Gate:
    """One validity check and what it observed."""

    name: str
    passed: bool
    observed: str
    requirement: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "requirement": self.requirement,
            "detail": self.detail,
        }


def random_baseline_gate(view: dict[str, Any]) -> Gate:
    """The random baseline's 95% CI must contain 0.50.

    This is the gate run #1 would have failed. Its CI was reported around a
    point estimate of 0.746 on 7 positives.

    A missing or undefined interval fails the gate rather than skipping it. If
    the apparatus could not measure chance, it has not been shown to measure
    anything.
    """
    signals = view.get("signals", {})
    entry = signals.get(RANDOM_SIGNAL)
    if not entry:
        return Gate(
            name="random_baseline_ci_contains_chance",
            passed=False,
            observed="absent",
            requirement="t_random 95% CI must contain 0.50",
            detail=f"{RANDOM_SIGNAL} is not in the signal table, so chance was never measured",
        )
    ci = entry.get("auroc", {})
    point = ci.get("point")
    low = ci.get("ci_low")
    high = ci.get("ci_high")
    if point is None or low is None or high is None:
        return Gate(
            name="random_baseline_ci_contains_chance",
            passed=False,
            observed="undefined",
            requirement="t_random 95% CI must contain 0.50",
            detail="the random baseline has no defined confidence interval",
        )
    contains = float(low) <= 0.50 <= float(high)
    return Gate(
        name="random_baseline_ci_contains_chance",
        passed=contains,
        observed=f"AUROC {float(point):.3f} [{float(low):.3f}, {float(high):.3f}]",
        requirement="t_random 95% CI must contain 0.50",
        detail=(
            "the random baseline brackets chance, so the bootstrap interval is "
            "wide enough to be believed"
            if contains
            else "the random baseline excludes 0.50, so no AUROC in this table is "
            "interpretable; this is the run #1 failure mode"
        ),
    )


def class_balance_gate(view: dict[str, Any]) -> Gate:
    """At least MIN_PER_CLASS positives and MIN_PER_CLASS negatives."""
    n_incorrect = int(view.get("n_incorrect", 0))
    n_correct = int(view.get("n_correct", 0))
    ok = n_incorrect >= MIN_PER_CLASS and n_correct >= MIN_PER_CLASS
    return Gate(
        name="minimum_rows_per_class",
        passed=ok,
        observed=f"{n_incorrect} incorrect, {n_correct} correct",
        requirement=f">= {MIN_PER_CLASS} rows in each class",
        detail=(
            "both classes are large enough for the ranking to be powered"
            if ok
            else "at least one class is below the floor, so the ranking is "
            "underpowered and the ordering of signals should not be relied on"
        ),
    )


def abstention_gate(n_abstentions: int, n_scored: int) -> Gate:
    """Abstention rate must be under MAX_ABSTENTION_RATE.

    Reported either way, per the brief. Run #1 abstained on 25 of 100 questions;
    run #2 removed the abstention instruction from the system prompt.
    """
    rate = (n_abstentions / n_scored) if n_scored else float("nan")
    ok = math.isfinite(rate) and rate < MAX_ABSTENTION_RATE
    return Gate(
        name="abstention_rate_below_ceiling",
        passed=ok,
        observed=f"{n_abstentions}/{n_scored} = {rate:.3f}" if n_scored else "no rows",
        requirement=f"< {MAX_ABSTENTION_RATE:.2f} of scored rows",
        detail=(
            "few enough refusals that the table is not measuring refusal detection"
            if ok
            else "refusals are frequent enough to inflate every AUROC, because an "
            "empty answer is trivially predictable from any of these signals"
        ),
    )


def human_label_gate(coverage: float | None) -> Gate:
    """Human-label coverage of the validation sample must clear the floor.

    `coverage` is labelled validation rows over validation rows (None when the
    file is absent or unreadable). Records failure rather than raising, per the
    module convention: the missing human labels are a visible gate failure, not
    a paragraph in LIMITATIONS. Today this gate fails honestly at 0.0.
    """
    ok = coverage is not None and math.isfinite(coverage) and coverage >= MIN_HUMAN_LABEL_COVERAGE
    observed = "validation file absent" if coverage is None else f"coverage {coverage:.3f}"
    return Gate(
        name="human_label_coverage",
        passed=ok,
        observed=observed,
        requirement=f">= {MIN_HUMAN_LABEL_COVERAGE:.2f} of validation rows labelled",
        detail=(
            "human labels bound the machine label error, which bounds what any "
            "AUROC against those labels can mean"
            if ok
            else "no human has verified any label, so the label set's correctness "
            "is unmeasured; fill data/human_validation_sample.csv (docs/HUMAN_LABELING.md)"
        ),
    )


def evaluate_gates(
    view: dict[str, Any],
    *,
    n_abstentions: int,
    n_scored: int,
    human_label_coverage: float | None = None,
) -> dict[str, Any]:
    """Run every gate and summarize. Never raises.

    `all_passed` is what the README branches on. When it is false the results
    section must open with VALIDITY FAILED and must not publish a ranking.
    """
    gates = [
        random_baseline_gate(view),
        class_balance_gate(view),
        abstention_gate(n_abstentions, n_scored),
        human_label_gate(human_label_coverage),
    ]
    failed = [g.name for g in gates if not g.passed]
    return {
        "all_passed": not failed,
        "failed": failed,
        "gates": [g.as_dict() for g in gates],
        "ranking_publishable": not failed,
    }


def assert_frozen_analysis_set(
    frame: Any,
    names: list[str],
    *,
    view_name: str,
) -> dict[str, Any]:
    """Assert every signal is scored on the identical row set (D12).

    The concern is a silent per-signal subset. Signals differ in how many rows
    they can produce a value for — an OpenAI-shaped gateway returning no
    logprobs leaves family A empty, a generation failure leaves family B empty —
    and the wrong way to handle that is to let each signal quietly drop its own
    missing rows and then compare the resulting AUROCs as though they described
    the same questions.

    So the row set is fixed once per view and every signal is evaluated against
    exactly it. Missing values inside that fixed set are recorded as
    `missing_n`, and any signal whose usable rows differ from the frozen set is
    reported here rather than hidden. This raises on a structural violation
    (duplicate or missing qids) because that is a bug, not a result.
    """
    qids = [str(q) for q in frame["qid"].tolist()]
    if len(qids) != len(set(qids)):
        raise AssertionError(f"{view_name}: the analysis set contains duplicate qids")

    missing_columns = [n for n in names if n not in frame.columns]
    if missing_columns:
        raise AssertionError(
            f"{view_name}: signals absent from the analysis frame: {missing_columns}"
        )

    # Every signal column has one entry per frozen row. This is the invariant
    # that makes the AUROCs comparable.
    for name in names:
        if len(frame[name]) != len(qids):
            raise AssertionError(
                f"{view_name}: signal {name} has {len(frame[name])} values for " f"{len(qids)} rows"
            )

    return {
        "view": view_name,
        "n_rows": len(qids),
        "qid_digest": _digest(qids),
        "n_signals": len(names),
        "note": (
            "every signal is scored on this exact row set; per-signal gaps are "
            "recorded as missing_n rather than by dropping rows"
        ),
    }


def _digest(qids: list[str]) -> str:
    """Order-independent fingerprint of a row set.

    Written into results.json so that two views, or two runs, can be checked for
    "same rows" without shipping the whole qid list twice.
    """
    import hashlib

    joined = "\n".join(sorted(qids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]
