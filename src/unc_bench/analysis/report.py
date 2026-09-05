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
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from unc_bench.analysis.extended import (
    cost_table,
    length_confound,
    logreg_combination,
    per_dataset_auroc,
    significance_table,
    verbal_confidence_health,
)
from unc_bench.analysis.metrics import (
    apply_platt,
    bootstrap_auroc_ci,
    bootstrap_average_precision_ci,
    expected_calibration_error,
    fit_platt,
    risk_coverage,
    spearman_matrix,
    usable_mask,
)
from unc_bench.analysis.validity import assert_frozen_analysis_set, evaluate_gates
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

    # Carry the source dataset and the greedy answer through, for the
    # per-dataset breakdown (D9) and the answer-length confound (D10). Joined
    # here rather than recomputed so the per-dataset slices are guaranteed to be
    # subsets of the same frozen row set as the pooled table.
    dataset_frame = read_checkpoint(paths.dataset)
    if dataset_frame is not None and "dataset" in dataset_frame.columns:
        merged = merged.merge(
            dataset_frame[["qid", "dataset"]], on="qid", how="left", validate="one_to_one"
        )
    if dataset_frame is not None and "question" in dataset_frame.columns:
        merged = merged.merge(
            dataset_frame[["qid", "question"]], on="qid", how="left", validate="one_to_one"
        )
    if generations is not None and "greedy_answer" in generations.columns:
        merged = merged.merge(
            generations[["qid", "greedy_answer"]], on="qid", how="left", validate="one_to_one"
        )
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
    frozen: dict[str, Any] = {}
    for view_name, frame in views.items():
        # D12: fix the row set before any signal is scored, and assert that every
        # signal is scored against exactly it.
        frozen[view_name] = assert_frozen_analysis_set(frame, names, view_name=view_name)
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

    dataset_meta_path = cfg.paths.artifacts_dir / "dataset_meta.json"
    dataset_meta: dict[str, Any] = {}
    if dataset_meta_path.exists():
        try:
            dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            dataset_meta = {}
    gold_leakage = {
        "dropped": sum(int(v.get("gold_leakage_dropped", 0)) for v in dataset_meta.values())
        if dataset_meta
        else None,
        "inspected": sum(int(v.get("gold_leakage_inspected", 0)) for v in dataset_meta.values())
        if dataset_meta
        else None,
        "per_dataset": dataset_meta,
        "method": "normalized gold alias as contiguous token subsequence of the question",
    }

    # Label quality from the human validation sample. Never breaks analysis:
    # a missing or empty human column is the shipped state, reported as
    # coverage 0.0 with a reason rather than as a number.
    label_quality = _label_quality(path=cfg.paths.human_validation_csv)

    # D15: the gates are evaluated on the primary view, which is the one the
    # README quotes. A failure is recorded, not raised: the failure is the result.
    gates = evaluate_gates(
        per_view["primary"],
        n_abstentions=n_abstain,
        n_scored=int(len(scored)),
        human_label_coverage=label_quality.get("coverage"),
    )
    # D7: cost per signal, using this run's own measured timings.
    costs = cost_table(per_view["primary"]["signals"], timings, cfg)

    # D6: the N-ablation is produced by its own stage because it needs the NLI
    # model. Folded in here when present so results.json stays the single file
    # the README and the figures both read.
    ablation_path = cfg.paths.artifacts_dir / "ablation.json"
    ablation: dict[str, Any] | None = None
    if ablation_path.exists():
        try:
            ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ablation = None

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
            "gold_leakage": gold_leakage,
        },
        "labels": {
            "counts": {str(k): int(v) for k, v in merged["label"].value_counts().items()},
            "sources": {str(k): int(v) for k, v in merged["source"].value_counts().items()},
            "n_ambiguous_dropped": n_ambiguous,
            "n_abstentions": n_abstain,
            "kappa": label_meta.get("kappa", {"available": False, "reason": "label stage not run"}),
            "judge_parse_failures": label_meta.get("judge_parse_failures"),
            "heuristic_fallback_rows": label_meta.get("heuristic_fallback_rows"),
            "label_quality": label_quality,
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
        "validity_gates": gates,
        "frozen_analysis_set": frozen,
        "cost": costs,
        "ablation": ablation,
        "views": per_view,
        "timings": timings,
        "seeds": {
            "dataset_seed": cfg.dataset_seed,
            "greedy_seed": cfg.greedy.seed,
            "sampling_seed_base": cfg.sampling.seed_base,
            "bootstrap_seed": cfg.analysis.bootstrap_seed,
            "logreg_seed": cfg.analysis.logreg_seed,
            "split_seed": cfg.split.seed,
        },
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


#: Signals whose raw value is already a probability in [0,1], so an ECE can be
#: computed on them before any recalibration. Everything else (logprobs,
#: entropies, counts, lengths) has no pre-Platt ECE that means anything.
PROBABILITY_VALUED = ("c_p_true_plain", "c_p_true_with_samples", "c_verbal_confidence")


def _label_quality(path: Path) -> dict[str, Any]:
    """Human-label coverage and agreement summary for `results.json`.

    Reads the validation CSV through the same scorer the CLI uses, so the
    numbers here and in `unc-bench human-agreement` cannot disagree. Any
    failure — absent file, empty column, unreadable rows — yields coverage
    None with a reason rather than raising: the analysis stage must never fail
    because a human has not labelled yet.
    """
    try:
        from unc_bench.analysis.human_agreement import build_report
    except ImportError as exc:
        return {
            "human_labels_present": False,
            "coverage": None,
            "reason": f"human-agreement module unavailable: {exc}",
        }
    try:
        validation = build_report(path)
    except (OSError, KeyError, ValueError, UnicodeDecodeError) as exc:
        return {
            "human_labels_present": False,
            "coverage": None,
            "reason": f"validation file unreadable: {exc}",
        }
    if validation.n_rows == 0 or validation.n_labelled == 0:
        return {
            "human_labels_present": False,
            "coverage": 0.0 if validation.n_rows else None,
            "n_rows": validation.n_rows,
            "n_labelled": 0,
            "reason": "no human labels present",
        }
    machine = next((a for a in validation.agreements if a.machine_column == "machine_label"), None)
    return {
        "human_labels_present": True,
        "coverage": validation.n_labelled / validation.n_rows,
        "n_rows": validation.n_rows,
        "n_labelled": validation.n_labelled,
        "machine_agreement_rate": (
            machine.n_agree / machine.n_compared if machine and machine.n_compared else None
        ),
        "machine_kappa": machine.kappa if machine else None,
        "machine_kappa_ci": [machine.kappa_ci_low, machine.kappa_ci_high] if machine else None,
        "oracle_ceiling": validation.oracle_ceiling,
    }


def _pre_platt_calibration(
    name: str,
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[np.bool_],
    eval_mask: npt.NDArray[np.bool_],
    cfg: Config,
) -> dict[str, Any]:
    """ECE before recalibration, for the probability-valued signals only (D5).

    The stored column is oriented, meaning "higher = more likely wrong", and for
    these three signals that orientation was produced by negating a confidence.
    So the model's own asserted probability of being wrong is 1 + oriented_value
    — the negation undone, then complemented. That is the number whose
    calibration the reader cares about: it is what the model claimed.

    Non-probability signals return null rather than a number. Min-max scaling a
    logprob column into [0,1] would produce an ECE, and it would be an ECE of the
    scaling choice rather than of the signal.
    """
    if name not in PROBABILITY_VALUED:
        return {"ece": None, "calibration": None, "is_probability_valued": False}
    asserted_wrong = 1.0 + scores
    finite = np.isfinite(asserted_wrong)
    use = eval_mask & finite
    if not int(np.count_nonzero(use)):
        return {"ece": None, "calibration": None, "is_probability_valued": True}
    cal = expected_calibration_error(
        np.clip(asserted_wrong[use], 0.0, 1.0), y[use], bins=cfg.analysis.ece_bins
    )
    return {"ece": cal.ece, "calibration": cal.as_dict(), "is_probability_valued": True}


def _cluster_ids(frame: pd.DataFrame, cfg: Config) -> npt.NDArray[np.int64] | None:
    """Question-text cluster ids for the cluster bootstrap, or None.

    Factorized normalized question text over the frozen view rows. Returned
    only when `analysis.cluster_bootstrap` is on; otherwise None, which keeps
    every bootstrap on the row-level estimator the committed intervals used.
    """
    if not cfg.analysis.cluster_bootstrap or "question" not in frame.columns:
        return None
    from unc_bench.normalize import normalize_answer

    codes = (
        frame["question"]
        .astype(str)
        .map(normalize_answer)
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int64)
    )
    return np.asarray(codes, dtype=np.int64)


def _analyze_view(frame: pd.DataFrame, names: list[str], cfg: Config) -> dict[str, Any]:
    """AUROC table, risk-coverage, calibration and correlations for one view."""
    n = int(len(frame))
    y = (frame["label"] == LABEL_INCORRECT).to_numpy(dtype=bool)
    base_rate = float(np.mean(y)) if n else float("nan")
    qids = [str(q) for q in frame["qid"].tolist()]
    is_train = _train_mask(qids, cfg) if n else np.zeros(0, dtype=bool)

    columns = {name: frame[name].to_numpy(dtype=np.float64) for name in names}
    clusters = _cluster_ids(frame, cfg)

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
            cluster_ids=clusters,
        )
        ap = bootstrap_average_precision_ci(
            scores,
            y,
            resamples=cfg.analysis.bootstrap_resamples,
            seed=cfg.analysis.bootstrap_seed,
            level=cfg.analysis.ci_level,
            cluster_ids=clusters,
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

        # ECE BEFORE recalibration (D5), on the same held-out rows, so the two
        # numbers are comparable. Only signals already on a probability scale
        # have a meaningful pre-Platt ECE: a raw logprob is not a probability and
        # squeezing one into [0,1] to score it would invent the calibration being
        # measured. Those signals get null and are named as such.
        pre_platt = _pre_platt_calibration(name, scores, y, eval_mask, cfg)

        signals_out[name] = {
            "family": spec.family,
            "orientation": spec.orientation,
            "description": spec.description,
            "auroc": ci.as_dict(),
            "auprc": ap.as_dict(),
            "auprc_baseline": base_rate,
            "usable_n": int(np.count_nonzero(usable_mask(scores))),
            "missing_n": int(np.count_nonzero(~usable_mask(scores))),
            "aurc": rc.aurc,
            "coverage_at_target_accuracy": rc.coverage_at_target,
            "platt": {"a": a, "b": b, "train_n": int(is_train.sum())},
            "ece": calibration.ece,
            "ece_after_platt": calibration.ece,
            "ece_before_platt": pre_platt["ece"],
            "is_probability_valued": pre_platt["is_probability_valued"],
            "calibration": calibration.as_dict(),
            "calibration_before_platt": pre_platt["calibration"],
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
        # D3, D8, D9, D10: computed on the same frozen rows as the table above.
        "significance": significance_table(columns, y, ranked, cfg, clusters),
        "per_dataset": per_dataset_auroc(frame, names, y, cfg, clusters),
        "cluster_bootstrap": {
            "enabled": clusters is not None,
            "n_clusters": int(len(set(clusters.tolist()))) if clusters is not None else 0,
            "key": "normalized question text",
        },
        "length_confound": length_confound(columns, y, ranked),
        "combination": logreg_combination(columns, y, ranked, cfg),
        "verbal_confidence_health": verbal_confidence_health(frame),
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


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null, recursively.

    `json.dumps` emits a bare `NaN` token by default. Python's own loader
    accepts it, so this file round-trips locally and looks fine, but `NaN` is
    not in the JSON grammar: every strict parser rejects it, which includes
    `jq`, JavaScript's `JSON.parse` and most CI validators. A results file that
    only the language that wrote it can read is not a portable artifact.

    `null` rather than a sentinel number. NaN here means "not measured" — a
    signal with no usable rows, a CI that could not be estimated — and any
    numeric stand-in would be silently averaged into something by a reader.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_results(cfg: Config) -> dict[str, Any]:
    """Build the report and write it to `paths.results_json`."""
    results = build_results(cfg)
    path = cfg.paths.results_json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"[analyze] wrote {path}", flush=True)
    return results
