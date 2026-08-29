"""Stage 5: assemble results.json from the signal and label artifacts.

Two label views are computed, because the choice is not neutral. `primary`
excludes abstentions; `with_abstentions` counts them as errors. A refusal is
trivially predictable from any of these signals — an empty answer has a
distinctive logprob profile and five identical refusals have zero semantic
entropy — so scoring refusals as errors inflates every AUROC in the table. The
primary view is the one the README quotes and the other is reported beside it so
the size of that effect is visible rather than argued about.

Ambiguous items are dropped from both views and counted.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from unc_bench.analysis.metrics import (
    apply_platt,
    bootstrap_auroc_ci,
    expected_calibration_error,
    fit_platt,
    risk_coverage,
    spearman_matrix,
    usable_mask,
)
from unc_bench.config import Config
from unc_bench.signals.base import FAMILY_LABELS, get_spec, signal_names
from unc_bench.stages.common import StagePaths, read_checkpoint
from unc_bench.stages.score_signals import merged_signals
from unc_bench.types import LABEL_ABSTAIN, LABEL_AMBIGUOUS, LABEL_INCORRECT


def _train_mask(qids: list[str], cfg: Config) -> npt.NDArray[np.bool_]:
    """Deterministic train/eval split, used only to fit recalibrators.

    Keyed on sorted qids so the split does not depend on row order, matching the
    convention in the dataset builders.
    """
    ordered = sorted(qids)
    n_train = int(round(cfg.split.train_fraction * len(ordered)))
    rng = np.random.default_rng(cfg.split.seed)
    picked = set()
    if n_train > 0 and ordered:
        idx = rng.choice(len(ordered), size=min(n_train, len(ordered)), replace=False)
        picked = {ordered[int(i)] for i in idx}
    return np.array([q in picked for q in qids], dtype=bool)


def build_results(cfg: Config) -> dict[str, Any]:
    """Every number the README quotes, in one dictionary."""
    paths = StagePaths.of(cfg)
    signals = merged_signals(cfg)
    labels = read_checkpoint(paths.labels)
    if labels is None:
        raise FileNotFoundError(f"{paths.labels} is absent; run label first")
    generations = read_checkpoint(paths.generations)

    merged = signals.merge(labels, on="qid", how="inner", validate="one_to_one")
    total_rows = int(len(merged))
    n_ambiguous = int((merged["label"] == LABEL_AMBIGUOUS).sum())
    n_abstain = int((merged["label"] == LABEL_ABSTAIN).sum())
    scored = merged.loc[merged["label"] != LABEL_AMBIGUOUS].reset_index(drop=True)

    views = {
        "primary": scored.loc[scored["label"] != LABEL_ABSTAIN].reset_index(drop=True),
        "with_abstentions": scored,
    }

    names = [n for n in signal_names() if n in merged.columns]
    per_view: dict[str, Any] = {}
    for view_name, frame in views.items():
        per_view[view_name] = _analyze_view(frame, names, cfg)

    label_meta_path = cfg.paths.artifacts_dir / "label_meta.json"
    label_meta: dict[str, Any] = {}
    if label_meta_path.exists():
        try:
            label_meta = json.loads(label_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            label_meta = {}

    timings: dict[str, Any] = {}
    if paths.timings.exists():
        try:
            timings = json.loads(paths.timings.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timings = {}

    return {
        "run_name": cfg.run_name,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "is_pilot": cfg.is_pilot,
        "model_under_test": {
            "name": cfg.model_under_test.name,
            "backend": cfg.model_under_test.backend,
            "dtype": cfg.model_under_test.dtype,
            "max_new_tokens": cfg.model_under_test.max_new_tokens,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.machine(),
        },
        "dataset": {
            "mix": {
                "popqa": cfg.dataset_mix.popqa,
                "triviaqa": cfg.dataset_mix.triviaqa,
                "simpleqa": cfg.dataset_mix.simpleqa,
            },
            "seed": cfg.dataset_seed,
            "rows_with_labels": total_rows,
            "generated_rows": 0 if generations is None else int(len(generations)),
        },
        "labels": {
            "counts": {str(k): int(v) for k, v in merged["label"].value_counts().items()},
            "sources": {str(k): int(v) for k, v in merged["source"].value_counts().items()},
            "n_ambiguous_dropped": n_ambiguous,
            "n_abstentions": n_abstain,
            "kappa": label_meta.get("kappa", {"available": False, "reason": "label stage not run"}),
            "judge_parse_failures": label_meta.get("judge_parse_failures"),
            "heuristic_fallback_rows": label_meta.get("heuristic_fallback_rows"),
        },
        "analysis_config": {
            "bootstrap_resamples": cfg.analysis.bootstrap_resamples,
            "bootstrap_seed": cfg.analysis.bootstrap_seed,
            "ci_level": cfg.analysis.ci_level,
            "ece_bins": cfg.analysis.ece_bins,
            "target_accuracy": cfg.analysis.target_accuracy,
            "train_fraction": cfg.split.train_fraction,
            "min_meaningful_auroc_gap": cfg.analysis.min_meaningful_auroc_gap,
        },
        "views": per_view,
        "timings": timings,
        "signal_catalog": {
            name: {
                "family": get_spec(name).family,
                "family_label": FAMILY_LABELS[get_spec(name).family],
                "orientation": get_spec(name).orientation,
                "description": get_spec(name).description,
            }
            for name in names
        },
    }


def _analyze_view(frame: pd.DataFrame, names: list[str], cfg: Config) -> dict[str, Any]:
    """AUROC table, risk-coverage, calibration and correlations for one view."""
    n = int(len(frame))
    y = (frame["label"] == LABEL_INCORRECT).to_numpy(dtype=bool)
    base_rate = float(np.mean(y)) if n else float("nan")
    qids = [str(q) for q in frame["qid"].tolist()]
    is_train = _train_mask(qids, cfg) if n else np.zeros(0, dtype=bool)

    columns = {name: frame[name].to_numpy(dtype=np.float64) for name in names}

    signals_out: dict[str, Any] = {}
    for name in names:
        scores = columns[name]
        spec = get_spec(name)
        ci = bootstrap_auroc_ci(
            scores,
            y,
            resamples=cfg.analysis.bootstrap_resamples,
            seed=cfg.analysis.bootstrap_seed,
            level=cfg.analysis.ci_level,
        )
        rc = risk_coverage(scores, y, target_accuracy=cfg.analysis.target_accuracy)

        # Platt map fitted on TRAIN only, ECE reported on the held-out rest.
        # Fitting and evaluating on the same rows would report the fit quality of
        # a two-parameter model, not the calibration of the signal.
        eval_mask = ~is_train
        a, b = fit_platt(scores[is_train], y[is_train]) if int(is_train.sum()) else (np.nan, np.nan)
        probs = apply_platt(scores, a, b)
        calibration = expected_calibration_error(
            probs[eval_mask], y[eval_mask], bins=cfg.analysis.ece_bins
        )

        signals_out[name] = {
            "family": spec.family,
            "orientation": spec.orientation,
            "description": spec.description,
            "auroc": ci.as_dict(),
            "usable_n": int(np.count_nonzero(usable_mask(scores))),
            "missing_n": int(np.count_nonzero(~usable_mask(scores))),
            "aurc": rc.aurc,
            "coverage_at_target_accuracy": rc.coverage_at_target,
            "platt": {"a": a, "b": b, "train_n": int(is_train.sum())},
            "ece": calibration.ece,
            "calibration": calibration.as_dict(),
            "risk_coverage": {"coverage": rc.coverage, "risk": rc.risk},
        }

    ranked = sorted(
        (nm for nm in names if np.isfinite(signals_out[nm]["auroc"]["point"])),
        key=lambda nm: signals_out[nm]["auroc"]["point"],
        reverse=True,
    )
    verdict = _verdict(ranked, signals_out, cfg)
    corr_names, corr = spearman_matrix(columns)

    return {
        "n": n,
        "base_rate_incorrect": base_rate,
        "n_incorrect": int(np.count_nonzero(y)),
        "n_correct": int(n - np.count_nonzero(y)),
        "signals": signals_out,
        "ranking": ranked,
        "verdict": verdict,
        "correlation": {
            "names": corr_names,
            "spearman": [[None if not np.isfinite(v) else float(v) for v in row] for row in corr],
        },
    }


def _verdict(ranked: list[str], signals: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Whether the top signal is meaningfully ahead of the field.

    Two checks, both of which have to pass before anything is called a winner.
    The CIs must not overlap, and the gap must exceed
    `min_meaningful_auroc_gap`. At n=150 a bootstrap CI is roughly ±0.07 wide, so
    two signals half a point apart are indistinguishable and saying otherwise
    would be the single most likely way to publish a wrong conclusion here.
    """
    if len(ranked) < 2:
        return {"winner": None, "reason": "fewer than two signals have a defined AUROC"}
    best, second = ranked[0], ranked[1]
    best_ci = signals[best]["auroc"]
    second_ci = signals[second]["auroc"]
    gap = float(best_ci["point"]) - float(second_ci["point"])
    disjoint = float(best_ci["ci_low"]) > float(second_ci["ci_high"])
    meaningful = gap >= cfg.analysis.min_meaningful_auroc_gap
    if disjoint and meaningful:
        return {
            "winner": best,
            "runner_up": second,
            "gap": gap,
            "reason": "top CI lies entirely above the runner-up's",
        }
    return {
        "winner": None,
        "top_signal": best,
        "runner_up": second,
        "gap": gap,
        "ci_overlap": not disjoint,
        "reason": (
            f"no winner: {best} leads {second} by {gap:.3f} AUROC, "
            f"{'CIs overlap' if not disjoint else 'gap below the meaningful threshold'} "
            f"at n={best_ci['n']}"
        ),
    }


def write_results(cfg: Config) -> dict[str, Any]:
    """Build the report and write it to `paths.results_json`."""
    results = build_results(cfg)
    path = cfg.paths.results_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[analyze] wrote {path}", flush=True)
    return results
