"""The autonomous decision that replaces asking a human whether to proceed.

The pilot measures two things and both gate the full run. Seconds per question
decides how many questions the remaining budget affords. The error rate decides
whether the question set is usable at all: near 0% and there are almost no
positives to predict, so every AUROC is an interval rather than an estimate; near
100% and there are almost no negatives, same problem from the other side. The
configured band is 25-65%.

Out of band, the recommendation is to change the dataset mix, once. Iterating on
the question set until the numbers look good is how a benchmark stops measuring
anything, so `max_iterations` is enforced and every iteration is recorded.
"""

from __future__ import annotations

import json
from typing import Any

from unc_bench.config import Config
from unc_bench.stages.common import StagePaths, read_checkpoint
from unc_bench.types import LABEL_ABSTAIN, LABEL_AMBIGUOUS, LABEL_INCORRECT


def evaluate(cfg: Config) -> dict[str, Any]:
    """Measure the pilot and return a recommendation."""
    paths = StagePaths.of(cfg)
    labels = read_checkpoint(paths.labels)
    if labels is None:
        raise FileNotFoundError(f"{paths.labels} is absent; run the pilot label stage first")

    counts = {str(k): int(v) for k, v in labels["label"].value_counts().items()}
    scored = labels.loc[labels["label"] != LABEL_AMBIGUOUS]
    primary = scored.loc[scored["label"] != LABEL_ABSTAIN]
    n_primary = int(len(primary))
    n_incorrect = int((primary["label"] == LABEL_INCORRECT).sum())
    error_rate = n_incorrect / n_primary if n_primary else float("nan")

    timings: dict[str, Any] = {}
    if paths.timings.exists():
        try:
            timings = json.loads(paths.timings.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timings = {}
    generate_timing = timings.get("generate", {})
    seconds_per_question = float(generate_timing.get("seconds_per_item", float("nan")))

    low = cfg.pilot_gate.error_rate_low
    high = cfg.pilot_gate.error_rate_high
    in_band = bool(n_primary and low <= error_rate <= high)

    if not n_primary:
        recommendation = "abort: no non-abstained, non-ambiguous rows to measure"
    elif in_band:
        recommendation = "proceed: error rate is inside the usable band"
    elif error_rate < low:
        recommendation = (
            "adjust dataset mix toward harder questions: the error rate is below the band, "
            "so there are too few positives to estimate an AUROC"
        )
    else:
        recommendation = (
            "adjust dataset mix toward easier questions: the error rate is above the band, "
            "so there are too few negatives to estimate an AUROC"
        )

    payload = {
        "run_name": cfg.run_name,
        "label_counts": counts,
        "n_scored": int(len(scored)),
        "n_primary": n_primary,
        "n_incorrect": n_incorrect,
        "error_rate": error_rate,
        "band": {"low": low, "high": high},
        "in_band": in_band,
        "seconds_per_question": seconds_per_question,
        "abstention_rate": (
            counts.get(LABEL_ABSTAIN, 0) / int(len(labels)) if len(labels) else float("nan")
        ),
        "ambiguous_rate": (
            counts.get(LABEL_AMBIGUOUS, 0) / int(len(labels)) if len(labels) else float("nan")
        ),
        "recommendation": recommendation,
        "max_iterations": cfg.pilot_gate.max_iterations,
    }
    paths.pilot_gate.parent.mkdir(parents=True, exist_ok=True)
    paths.pilot_gate.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("[pilot-gate] " + json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload
