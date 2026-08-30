"""The analyses that run #1 did not have: D3, D6, D7, D8, D9, D10.

Kept out of `report.py` so that module stays a readable assembly of the results
dictionary. Everything here takes a frozen analysis frame and returns plain
dictionaries destined for results.json.

Each function is written to degrade to a recorded "not computed" rather than
raise. At n=120 several of these can legitimately fail to produce a number — a
per-dataset slice can be one-class, a family-B ablation needs the raw samples —
and a missing entry with a stated reason is a usable result while a traceback in
the middle of the analysis stage is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from unc_bench.analysis.metrics import (
    auroc,
    average_precision,
    bootstrap_auroc_ci,
    bootstrap_average_precision_ci,
    holm_bonferroni,
    paired_bootstrap_auroc_diff,
    usable_mask,
)
from unc_bench.config import Config
from unc_bench.signals.base import get_spec

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


# --------------------------------------------------------------------------
# D3: pairwise significance against the top-ranked signal
# --------------------------------------------------------------------------


def significance_table(
    columns: dict[str, FloatArray],
    y: BoolArray,
    ranking: list[str],
    cfg: Config,
) -> dict[str, Any]:
    """Every signal compared against the top-ranked one, Holm-corrected.

    The reference is the rank-1 signal, so the table answers "is anything
    distinguishable from the leader", which is the question the README's verdict
    line needs. Comparing all 210 pairs would answer a question nobody asked and
    cost twenty times the Holm penalty.

    The test is a paired bootstrap on the AUROC difference, not DeLong. The
    reason is in `paired_bootstrap_auroc_diff`: these signals are heavily tied
    and DeLong's tie handling is the fragile part. The README names the test that
    was actually run.
    """
    if len(ranking) < 2:
        return {
            "available": False,
            "reason": "fewer than two signals have a defined AUROC",
            "test": None,
        }

    reference = ranking[0]
    others = list(ranking[1:])
    raw: list[dict[str, Any]] = []
    for name in others:
        delta, low, high, p, n_paired = paired_bootstrap_auroc_diff(
            columns[reference],
            columns[name],
            y,
            resamples=cfg.analysis.bootstrap_resamples,
            seed=cfg.analysis.bootstrap_seed,
            level=cfg.analysis.ci_level,
        )
        raw.append(
            {
                "name": name,
                "delta_vs_reference": delta,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "n_paired": n_paired,
            }
        )

    adjusted = holm_bonferroni([r["p_value"] for r in raw])
    alpha = cfg.analysis.holm_alpha
    for entry, adj in zip(raw, adjusted, strict=True):
        entry["p_value_holm"] = adj
        entry["significant_holm"] = bool(np.isfinite(adj) and adj <= alpha)

    n_sig = sum(1 for r in raw if r["significant_holm"])
    return {
        "available": True,
        "test": "paired stratified bootstrap on the AUROC difference",
        "test_detail": (
            f"{cfg.analysis.bootstrap_resamples} resamples, identical resample "
            f"indices applied to both signals, two-sided achieved significance "
            f"level, Holm-Bonferroni across {len(raw)} comparisons at "
            f"alpha={alpha}"
        ),
        "delong_substituted": True,
        "delong_substitution_reason": (
            "DeLong's analytic variance assumes untied scores; several signals "
            "here are integer-valued with heavy ties, so a paired bootstrap was "
            "used instead"
        ),
        "reference": reference,
        "alpha": alpha,
        "n_comparisons": len(raw),
        "n_significant_after_holm": n_sig,
        "comparisons": raw,
    }


# --------------------------------------------------------------------------
# D9: per-dataset breakdown
# --------------------------------------------------------------------------


#: Method string stored beside every per-dataset interval, so a reader can tell
#: at a glance which procedure produced it without cross-referencing prose.
PER_DATASET_CI_METHOD = (
    "stratified percentile bootstrap, resampled within the dataset subset only; "
    "the same procedure as the pooled table"
)


def per_dataset_auroc(
    frame: pd.DataFrame,
    names: list[str],
    y: BoolArray,
    cfg: Config | None = None,
) -> dict[str, Any]:
    """AUROC per signal within each source dataset, alongside the pooled table.

    Pooling TriviaQA and PopQA hides the possibility that a signal only works on
    one of them. It also hides a subtler thing worth naming: the two sources have
    different base rates in this run, and a signal that merely tracked "which
    dataset is this" would score above chance on the pooled set while scoring at
    chance inside each source. The per-dataset table is what exposes that.

    When `cfg` is supplied, each point estimate is accompanied by a stratified
    percentile bootstrap interval computed *within the dataset subset*: the
    positives and negatives of that subset are resampled among themselves, so
    the interval reflects that subset's own size and class balance and never
    borrows rows from the other dataset. This is the same estimator
    `bootstrap_auroc_ci` applies to the pooled table, called on a row mask, so
    the two intervals in the results file are produced by one procedure rather
    than by two that happen to be described with the same word.

    `cfg` is optional so that a caller with no config — the ablation stage, a
    test — still gets the point estimates. Without it the entry carries
    `ci_available: false` and a reason, rather than an interval from a different
    method silently standing in for the bootstrap.
    """
    if "dataset" not in frame.columns:
        return {"available": False, "reason": "the analysis frame carries no dataset column"}

    resamples = cfg.analysis.bootstrap_resamples if cfg is not None else 0
    out: dict[str, Any] = {
        "available": True,
        "ci_available": cfg is not None,
        "ci_method": PER_DATASET_CI_METHOD if cfg is not None else None,
        "ci_resamples": resamples,
        "ci_level": cfg.analysis.ci_level if cfg is not None else None,
        "ci_seed": cfg.analysis.bootstrap_seed if cfg is not None else None,
        "datasets": {},
    }
    if cfg is None:
        out["ci_reason"] = "no analysis config supplied, so no bootstrap was run"

    for source in sorted(str(d) for d in frame["dataset"].unique()):
        mask = (frame["dataset"] == source).to_numpy(dtype=bool)
        y_sub = y[mask]
        n_pos = int(np.count_nonzero(y_sub))
        n_neg = int(y_sub.size - n_pos)
        entry: dict[str, Any] = {
            "n": int(y_sub.size),
            "n_incorrect": n_pos,
            "n_correct": n_neg,
            "base_rate_incorrect": float(np.mean(y_sub)) if y_sub.size else None,
        }
        if n_pos == 0 or n_neg == 0:
            entry["signals"] = {}
            entry["note"] = (
                f"{source} is single-class in this run ({n_pos} incorrect, "
                f"{n_neg} correct), so no AUROC is defined inside it"
            )
        else:
            entry["signals"] = {
                name: _per_dataset_signal(frame[name].to_numpy(dtype=np.float64)[mask], y_sub, cfg)
                for name in names
            }
        out["datasets"][source] = entry
    return out


def _per_dataset_signal(
    scores: FloatArray,
    y_sub: BoolArray,
    cfg: Config | None,
) -> dict[str, Any]:
    """One signal on one dataset subset: point estimates, then the interval.

    The point estimates are computed unconditionally and are byte-identical to
    what this function returned before intervals existed, which is what lets the
    bootstrap be added without moving any published number.
    """
    entry: dict[str, Any] = {
        "auroc": auroc(scores, y_sub),
        "auprc": average_precision(scores, y_sub),
        "usable_n": int(np.count_nonzero(usable_mask(scores))),
    }
    if cfg is None:
        return entry

    ci = bootstrap_auroc_ci(
        scores,
        y_sub,
        resamples=cfg.analysis.bootstrap_resamples,
        seed=cfg.analysis.bootstrap_seed,
        level=cfg.analysis.ci_level,
    )
    ap = bootstrap_average_precision_ci(
        scores,
        y_sub,
        resamples=cfg.analysis.bootstrap_resamples,
        seed=cfg.analysis.bootstrap_seed,
        level=cfg.analysis.ci_level,
    )
    entry["auroc_ci"] = ci.as_dict()
    entry["auprc_ci"] = ap.as_dict()
    # Recorded rather than left to the reader to derive, because "does this
    # interval clear chance" is the one question the per-dataset table exists to
    # answer and a reader recomputing it from two floats can get the boundary
    # case wrong. None when the interval could not be estimated at all.
    if np.isfinite(ci.low) and np.isfinite(ci.high):
        entry["excludes_chance"] = bool(ci.low > 0.5 or ci.high < 0.5)
    else:
        entry["excludes_chance"] = None
    return entry


# --------------------------------------------------------------------------
# D10: answer-length confound
# --------------------------------------------------------------------------


def length_confound(
    columns: dict[str, FloatArray],
    y: BoolArray,
    ranking: list[str],
) -> dict[str, Any]:
    """Does the leading signal survive once answer length is controlled for?

    Answer length is a confound with teeth here. A 0.5B model that does not know
    a fact tends to ramble — several of run #2's wrong answers are truncated
    sentences like "Petra, also known as Hira Mar'a in Arabi" against a gold of
    "JORDAN" — while a known fact comes out as one or two tokens. So length
    correlates with correctness on its own, and a signal computed over token
    logprobs correlates with length almost by construction, because summing or
    averaging over more tokens changes the statistic.

    Two things are reported. First, the Spearman correlation between each signal
    and answer length, which says how entangled they are. Second, a stratified
    AUROC for the leading signal: rows are split at the median answer length and
    the signal's AUROC is recomputed inside each stratum. If the leader only
    works because it is reading length, its within-stratum AUROCs collapse
    toward chance while the pooled one stays high.
    """
    if "t_answer_length" not in columns:
        return {"available": False, "reason": "t_answer_length is not in the signal table"}
    if not ranking:
        return {"available": False, "reason": "no signal has a defined AUROC"}

    length = columns["t_answer_length"]
    from unc_bench.analysis.metrics import _spearman

    correlations = {
        name: _spearman(columns[name], length) for name in columns if name != "t_answer_length"
    }

    leader = ranking[0]
    keep = usable_mask(columns[leader]) & usable_mask(length)
    strata: dict[str, Any] = {}
    if int(np.count_nonzero(keep)) >= 4:
        lengths_kept = length[keep]
        median = float(np.median(lengths_kept))
        # Split at the median. Ties go to the "short" side, which is why the two
        # strata are not guaranteed equal in size and why n is reported per side.
        for label, sub in (
            ("at_or_below_median_length", lengths_kept <= median),
            ("above_median_length", lengths_kept > median),
        ):
            s = columns[leader][keep][sub]
            y_sub = y[keep][sub]
            n_pos = int(np.count_nonzero(y_sub))
            n_neg = int(y_sub.size - n_pos)
            strata[label] = {
                "n": int(y_sub.size),
                "n_incorrect": n_pos,
                "n_correct": n_neg,
                "auroc": (auroc(s, y_sub) if n_pos and n_neg else None),
            }
        strata["median_answer_length"] = median

    pooled = auroc(columns[leader], y)
    within = [
        v["auroc"] for k, v in strata.items() if isinstance(v, dict) and v.get("auroc") is not None
    ]
    survives: bool | None = None
    verdict = "not assessed: too few rows to stratify"
    if within:
        worst = min(within)
        # "Survives" means the signal still discriminates inside both length
        # strata, i.e. it is not merely a length proxy. The 0.55 floor is a
        # judgement call and is stated as one in the README.
        survives = bool(worst > 0.55)
        verdict = (
            f"{leader} keeps AUROC {worst:.3f} in its weaker length stratum against "
            f"{pooled:.3f} pooled, so it is not purely a length proxy"
            if survives
            else f"{leader} falls to AUROC {worst:.3f} in its weaker length stratum "
            f"against {pooled:.3f} pooled, so much of its apparent skill is "
            f"answer length"
        )

    return {
        "available": True,
        "leader": leader,
        "pooled_auroc": pooled,
        "spearman_with_answer_length": correlations,
        "strata": strata,
        "leader_survives_length_control": survives,
        "verdict": verdict,
        "stratum_auroc_floor": 0.55,
    }


# --------------------------------------------------------------------------
# D8: logistic regression over the two best signals
# --------------------------------------------------------------------------


def logreg_combination(
    columns: dict[str, FloatArray],
    y: BoolArray,
    ranking: list[str],
    cfg: Config,
) -> dict[str, Any]:
    """Cross-validated AUROC for the top two signals combined.

    The question is narrow and worth asking: the two leading signals may be
    measuring the same thing, in which case combining them buys nothing and the
    cheap one is the right choice. Folds are stratified so each holdout keeps the
    base rate, and the reported number is the AUROC of the pooled out-of-fold
    predictions rather than the mean of per-fold AUROCs. Averaging per-fold
    AUROCs at 24 rows a fold is dominated by fold-level noise.

    Features are standardized using training-fold statistics only. Fitting the
    scaler on all rows would leak the holdout's distribution into the model.
    """
    usable_ranking = [n for n in ranking if n in columns]
    if len(usable_ranking) < 2:
        return {"available": False, "reason": "fewer than two ranked signals available"}

    first = usable_ranking[0]
    second, redundant = _first_distinct_partner(first, usable_ranking[1:], columns)
    if second is None:
        return {
            "available": False,
            "reason": (
                "every other ranked signal is rank-identical to the leader, so "
                "there is no second signal to combine with"
            ),
            "leader": first,
            "rank_identical_to_leader": redundant,
        }
    keep = usable_mask(columns[first]) & usable_mask(columns[second])
    x = np.column_stack([columns[first][keep], columns[second][keep]])
    y_sub = y[keep]
    n = int(y_sub.size)
    n_pos = int(np.count_nonzero(y_sub))
    n_neg = n - n_pos
    folds = int(cfg.analysis.logreg_cv_folds)
    if n_pos < folds or n_neg < folds:
        return {
            "available": False,
            "reason": (
                f"{folds}-fold CV needs at least {folds} rows per class; got "
                f"{n_pos} incorrect and {n_neg} correct on the paired subset"
            ),
        }

    rng = np.random.default_rng(cfg.analysis.logreg_seed)
    # Stratified fold assignment: shuffle within class, then deal round-robin.
    assignment = np.empty(n, dtype=np.int64)
    for cls in (True, False):
        idx = np.flatnonzero(y_sub == cls)
        rng.shuffle(idx)
        assignment[idx] = np.arange(idx.size) % folds

    oof = np.full(n, np.nan, dtype=np.float64)
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        mu = x[train].mean(axis=0)
        sd = x[train].std(axis=0)
        sd[sd == 0.0] = 1.0
        x_train = (x[train] - mu) / sd
        x_test = (x[test] - mu) / sd
        weights = _fit_logistic(x_train, y_sub[train])
        oof[test] = _logistic_score(x_test, weights)

    combined = auroc(oof, y_sub)
    # Single-signal baselines recomputed on the SAME paired subset, so the
    # comparison is like-for-like rather than against the full-column AUROC.
    best_single_name, best_single = max(
        ((nm, auroc(columns[nm][keep], y_sub)) for nm in (first, second)),
        key=lambda pair: pair[1],
    )
    improvement = combined - best_single
    return {
        "available": True,
        "signals": [first, second],
        "cv_folds": folds,
        "seed": cfg.analysis.logreg_seed,
        "n": n,
        "n_incorrect": n_pos,
        "n_correct": n_neg,
        "cross_validated_auroc": combined,
        "best_single_auroc_same_rows": best_single,
        "best_single_name": best_single_name,
        "improvement_over_best_single": improvement,
        "beats_best_single": bool(improvement > cfg.analysis.min_meaningful_auroc_gap),
        "skipped_as_rank_identical_to_leader": redundant,
        "note": (
            "out-of-fold predictions pooled then scored once; features "
            "standardized on training folds only"
        ),
    }


def _first_distinct_partner(
    leader: str,
    candidates: list[str],
    columns: dict[str, FloatArray],
    *,
    max_abs_spearman: float = 0.999,
) -> tuple[str | None, list[str]]:
    """The best-ranked signal that is not a monotone restatement of the leader.

    Necessary because several signals in this study are rank-identical by
    construction, not by coincidence. `b_distinct_count` and
    `b_distinct_fraction` differ by a constant divisor — the sample count is the
    same 5 for every row — so they have the same ranking, the same AUROC and the
    same information. Feeding both to a logistic regression asks whether a
    signal improves on itself, and the answer is a rounding error plus CV noise,
    which is what the first run of this function reported.

    So the partner is the highest-ranked signal whose absolute Spearman
    correlation with the leader is below `max_abs_spearman`. Perfect negative
    correlation counts as redundant too: a sign flip carries no extra
    information. Signals skipped this way are returned so the results file can
    name them rather than leaving the choice unexplained.
    """
    from unc_bench.analysis.metrics import _spearman

    skipped: list[str] = []
    for name in candidates:
        rho = _spearman(columns[leader], columns[name])
        if np.isfinite(rho) and abs(rho) >= max_abs_spearman:
            skipped.append(name)
            continue
        return name, skipped
    return None, skipped


def _fit_logistic(
    x: FloatArray, y: BoolArray, *, iterations: int = 200, lr: float = 0.1
) -> FloatArray:
    """Plain gradient-descent logistic regression with an intercept.

    Hand-rolled for the same reason as the metrics: two features and a hundred
    rows do not justify a scikit-learn dependency, and the convergence behaviour
    is easier to state than to configure. A small L2 term keeps the weights
    finite when a fold happens to be linearly separable, which does occur at
    this n.
    """
    design = np.column_stack([np.ones(x.shape[0]), x])
    target = y.astype(np.float64)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    l2 = 1e-3
    for _ in range(iterations):
        z = design @ weights
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        grad = design.T @ (p - target) / design.shape[0] + l2 * weights
        weights -= lr * grad
    return weights


def _logistic_score(x: FloatArray, weights: FloatArray) -> FloatArray:
    design = np.column_stack([np.ones(x.shape[0]), x])
    z = design @ weights
    scores: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    return scores


# --------------------------------------------------------------------------
# D2: verbalized-confidence parse health
# --------------------------------------------------------------------------


def verbal_confidence_health(frame: pd.DataFrame) -> dict[str, Any]:
    """Parse-failure rate and value distribution for verbalized confidence.

    The brief anticipated a near-constant signal, on the theory that a small
    model asked for a confidence number says "90" every time. Run #1's rows did
    not behave that way and neither should this be assumed: the distribution is
    reported so the reader can see whether the signal varies at all. If it is
    effectively constant then its AUROC is not a measurement of anything and that
    is the finding; if it varies, ranking it is legitimate.
    """
    if "c_verbal_confidence" not in frame.columns:
        return {"available": False, "reason": "c_verbal_confidence is not in the signal table"}

    raw = frame["c_verbal_confidence"].to_numpy(dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    n = int(raw.size)
    n_failed = int(n - finite.size)

    # The stored column is oriented (negated confidence). Recover the parsed
    # probability so the distribution reads in the units the model emitted.
    spec = get_spec("c_verbal_confidence")
    as_probability = -finite if spec.orientation == "confidence" else finite
    values, counts = np.unique(np.round(as_probability, 4), return_counts=True)
    distribution = {float(v): int(c) for v, c in zip(values, counts, strict=True)}
    modal_share = float(counts.max() / counts.sum()) if counts.size else float("nan")

    return {
        "available": True,
        "n": n,
        "n_parse_failures": n_failed,
        "parse_failure_rate": (n_failed / n) if n else float("nan"),
        "n_distinct_values": int(values.size),
        "value_distribution": distribution,
        "modal_value": float(values[int(np.argmax(counts))]) if values.size else None,
        "modal_share": modal_share,
        "mean": float(np.mean(as_probability)) if as_probability.size else None,
        "std": float(np.std(as_probability)) if as_probability.size else None,
        # A signal whose modal value covers almost everything cannot discriminate,
        # whatever AUROC falls out of it.
        "effectively_constant": bool(np.isfinite(modal_share) and modal_share >= 0.95),
    }


# --------------------------------------------------------------------------
# D7: cost per signal
# --------------------------------------------------------------------------


def cost_table(
    signals_out: dict[str, Any],
    timings: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    """Per-signal cost in extra model calls, tokens and wall-clock seconds.

    The study's actual question is which signal is worth its cost, so cost has
    to be a measured column rather than a footnote. Costs are expressed as
    multipliers over the one greedy answer every signal needs, and the
    wall-clock figures come from this run's own timings file rather than from an
    assumption.

    Family A is free: the logprobs arrive with the greedy answer. Family C costs
    one or two extra forward passes. Family B costs N sampled generations plus
    the NLI clustering pass, and it is the only family that needs a second model
    resident.
    """
    n_samples = cfg.sampling.n_samples
    generate = timings.get("generate", {})
    family_b = timings.get("family_b", {}) or timings.get("score_signals_b", {})
    per_question_s = generate.get("seconds_per_item")
    b_per_question_s = family_b.get("seconds_per_item")

    # One greedy answer is the unit. Everything else is quoted against it.
    families = {
        "A": {
            "extra_model_calls": 0.0,
            "note": "logprobs come back with the greedy answer, so family A is free",
        },
        "B": {
            "extra_model_calls": float(n_samples),
            "note": (
                f"{n_samples} sampled generations plus one NLI clustering pass "
                f"over the sample set"
            ),
        },
        "C": {
            "extra_model_calls": 1.0,
            "note": "one extra scoring pass for P(True) or one short generation",
        },
        "T": {
            "extra_model_calls": 0.0,
            "note": "computed from text already in hand",
        },
    }

    out: dict[str, Any] = {
        "available": True,
        "unit": "multiple of the single greedy answer every signal requires",
        "measured_generate_seconds_per_question": per_question_s,
        "measured_family_b_seconds_per_question": b_per_question_s,
        "n_samples": n_samples,
        "families": families,
        "signals": {},
    }
    for name in signals_out:
        family = get_spec(name).family
        extra = float(families[family]["extra_model_calls"])  # type: ignore[arg-type]
        # Wall clock attributable to the signal beyond the shared greedy answer.
        seconds = None
        if family == "B" and b_per_question_s is not None and per_question_s is not None:
            seconds = float(b_per_question_s) + float(per_question_s) * extra
        elif per_question_s is not None:
            seconds = float(per_question_s) * extra
        out["signals"][name] = {
            "family": family,
            "cost_multiplier": 1.0 + extra,
            "extra_model_calls_per_question": extra,
            "extra_seconds_per_question": seconds,
            "auroc": signals_out[name]["auroc"]["point"],
        }
    return out
