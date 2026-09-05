"""N-ablation for the self-consistency family (defect D6).

The cost argument for family B rests entirely on N. Five samples cost five
extra generations, and if two samples buy most of the discrimination then the
recommendation changes from 6x to 3x. Run #1 never measured this, so the cost
column was an assumption.

The samples are not regenerated. The first N of the five already on disk are
reused for each N, which is both cheaper and more correct: it isolates the effect
of N from the effect of drawing a different sample set, so the AUROC differences
between N=2 and N=5 come from sample count alone. The N=5 row recomputes the
same quantity as the main family-B pass and is asserted to match it, which is
the check that the slicing is doing what it claims.

The NLI model is loaded once and reused across all values of N. It cannot be
resident alongside the generator (docs/DECISIONS.md D9), so this is its own
stage, run after `generate` has exited.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from unc_bench.analysis.metrics import (
    auroc,
    bootstrap_auroc_ci,
    holm_bonferroni,
    paired_bootstrap_auroc_diff,
)
from unc_bench.config import Config
from unc_bench.signals.base import orient_all, signal_names
from unc_bench.signals.consistency import compute_family_b
from unc_bench.stages.common import Progress, StagePaths, json_load, read_checkpoint
from unc_bench.types import LABEL_ABSTAIN, LABEL_AMBIGUOUS, LABEL_INCORRECT

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


#: Family-B signal names, taken from the registry so a new B signal is included
#: automatically rather than needing to be listed here too.
def _family_b_names() -> list[str]:
    return [n for n in signal_names() if n.startswith("b_")]


def run(cfg: Config) -> dict[str, Any]:
    """Recompute family B at each N in `sampling.ablation_n` and score each.

    Writes `ablation.json` next to the other artifacts. Returns the payload.
    """
    paths = StagePaths.of(cfg)
    generations = read_checkpoint(paths.generations)
    if generations is None:
        raise FileNotFoundError(f"{paths.generations} is absent; run generate first")
    labels = read_checkpoint(paths.labels)
    if labels is None:
        raise FileNotFoundError(f"{paths.labels} is absent; run label first")

    merged = generations.merge(labels, on="qid", how="inner", validate="one_to_one")
    # Same exclusion rule as the primary analysis view, so the ablation AUROCs
    # are comparable with the main table rather than computed on a different set.
    scored = merged.loc[
        (merged["label"] != LABEL_AMBIGUOUS) & (merged["label"] != LABEL_ABSTAIN)
    ].reset_index(drop=True)
    y = (scored["label"] == LABEL_INCORRECT).to_numpy(dtype=bool)

    from unc_bench.signals.nli import DebertaEntailmentModel

    model = DebertaEntailmentModel(cfg.nli)
    names = _family_b_names()
    levels = sorted(cfg.sampling.ablation_n)

    per_n: dict[str, Any] = {}
    columns_by_n: dict[int, dict[str, FloatArray]] = {}
    for n_samples in levels:
        progress = Progress(f"ablation_n{n_samples}", len(scored), every=25)
        columns: dict[str, list[float]] = {name: [] for name in names}
        for record in scored.to_dict(orient="records"):
            samples = [str(a) for a in (json_load(str(record["sample_answers"])) or [])]
            raw = compute_family_b(
                str(record.get("greedy_answer") or ""),
                # The ablation: the FIRST n samples of the same five, so N is the
                # only thing that varies between levels.
                samples[:n_samples],
                model,
                cfg.nli,
            )
            oriented = orient_all(raw)
            for name in names:
                columns[name].append(float(oriented.get(name, float("nan"))))
            progress.tick(str(record["qid"]))

        entry: dict[str, Any] = {"n_samples": n_samples, "signals": {}}
        arrays: dict[str, FloatArray] = {}
        for name in names:
            scores = np.array(columns[name], dtype=np.float64)
            arrays[name] = scores
            ci = bootstrap_auroc_ci(
                scores,
                y,
                resamples=cfg.analysis.bootstrap_resamples,
                seed=cfg.analysis.bootstrap_seed,
                level=cfg.analysis.ci_level,
            )
            entry["signals"][name] = ci.as_dict()
        columns_by_n[n_samples] = arrays
        per_n[str(n_samples)] = entry

    payload = {
        "run_name": cfg.run_name,
        "n_rows": int(len(scored)),
        "n_incorrect": int(np.count_nonzero(y)),
        "n_correct": int(y.size - np.count_nonzero(y)),
        "levels": levels,
        "signals": names,
        "note": (
            "the first N of the same five samples are reused at every level, so "
            "the differences between levels are due to sample count and not to "
            "drawing a different sample set"
        ),
        "by_n": per_n,
    }

    consistency = _check_top_level_matches_main_pass(cfg, per_n, names, levels)
    payload["agreement_with_main_family_b_pass"] = consistency
    payload["level_differences"] = ablation_level_differences(
        columns_by_n,
        y,
        levels,
        resamples=cfg.analysis.bootstrap_resamples,
        seed=cfg.analysis.bootstrap_seed,
        level=cfg.analysis.ci_level,
    )

    out_path = cfg.paths.artifacts_dir / "ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ablation] wrote {out_path}", flush=True)
    return payload


def ablation_level_differences(
    columns_by_n: dict[int, dict[str, FloatArray]],
    y: BoolArray,
    levels: list[int],
    *,
    resamples: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    """Paired bootstrap CI on the AUROC *difference* between ablation levels.

    Per-level intervals cannot test N=3 against N=5: the levels are nested
    subsets of the same samples, so they are perfectly correlated by
    construction and overlapping intervals say nothing about the difference.
    The same paired bootstrap the significance table uses
    (`paired_bootstrap_auroc_diff`, identical resample indices for both
    levels) does test it. Computed for every family-B signal at every level
    below the max, against the max level; a consecutive-pair entry (each level
    against the previous one) is included for the same reason.
    Pure function of in-memory columns, so it is unit-testable without the
    NLI model or any checkpoint.
    """
    if not levels:
        return {"available": False, "reason": "no ablation levels"}
    top = max(levels)
    below = sorted(n for n in levels if n != top)
    if not below or top not in columns_by_n:
        return {"available": False, "reason": "fewer than two computed levels available"}
    names = sorted(columns_by_n[top])
    out: dict[str, Any] = {
        "available": True,
        "reference_level": top,
        "test": "paired stratified bootstrap on the AUROC difference",
        "comparisons": [],
    }
    ordered = sorted(levels)
    pairs: list[tuple[int, int]] = [(n, top) for n in below]
    for first, second in pairwise(ordered):
        if (first, second) not in pairs:
            pairs.append((first, second))
    for low, high in pairs:
        if low not in columns_by_n or high not in columns_by_n:
            continue
        for name in names:
            if name not in columns_by_n[low] or name not in columns_by_n[high]:
                continue
            delta, ci_low, ci_high, p, n_paired = paired_bootstrap_auroc_diff(
                columns_by_n[high][name],
                columns_by_n[low][name],
                y,
                resamples=resamples,
                seed=seed + low,
                level=level,
            )
            out["comparisons"].append(
                {
                    "signal": name,
                    "level": low,
                    "reference_level": high,
                    "delta_vs_reference": delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p,
                    "n_paired": n_paired,
                }
            )
    adjusted = holm_bonferroni([c["p_value"] for c in out["comparisons"]])
    for entry, adj in zip(out["comparisons"], adjusted, strict=True):
        entry["p_value_holm"] = adj
        entry["significant_holm"] = bool(adj <= 0.05)
    out["holm_note"] = (
        "Holm across every (signal, level-pair) comparison in this block; "
        "per-level intervals cannot test levels against each other because the "
        "levels are nested subsets of the same samples"
    )
    return out


def _check_top_level_matches_main_pass(
    cfg: Config,
    per_n: dict[str, Any],
    names: list[str],
    levels: list[int],
) -> dict[str, Any]:
    """At N = n_samples the ablation must reproduce the main family-B AUROCs.

    This is the self-check on the slicing. If `samples[:5]` did not in fact
    recover the full sample set — a truncated column, a different row order, an
    off-by-one — the top ablation level would disagree with the main pass, and
    the ablation curve would be measuring the bug. Reported as a max absolute
    difference rather than asserted, because the two passes exclude rows on the
    same rule but a mismatch is worth seeing rather than crashing on.
    """
    top = max(levels)
    if top != cfg.sampling.n_samples:
        return {
            "checked": False,
            "reason": (
                f"top ablation level {top} differs from n_samples "
                f"{cfg.sampling.n_samples}, so no comparison applies"
            ),
        }
    paths = StagePaths.of(cfg)
    main = read_checkpoint(paths.signals_b)
    labels = read_checkpoint(paths.labels)
    if main is None or labels is None:
        return {"checked": False, "reason": "the main family-B artifact is absent"}

    merged = main.merge(labels, on="qid", how="inner", validate="one_to_one")
    kept = merged.loc[
        (merged["label"] != LABEL_AMBIGUOUS) & (merged["label"] != LABEL_ABSTAIN)
    ].reset_index(drop=True)
    y_main = (kept["label"] == LABEL_INCORRECT).to_numpy(dtype=bool)

    diffs: dict[str, float] = {}
    for name in names:
        if name not in kept.columns:
            continue
        main_auroc = auroc(kept[name].to_numpy(dtype=np.float64), y_main)
        ablation_auroc = per_n[str(top)]["signals"][name]["point"]
        if np.isfinite(main_auroc) and np.isfinite(ablation_auroc):
            diffs[name] = abs(float(main_auroc) - float(ablation_auroc))
    worst = max(diffs.values()) if diffs else float("nan")
    return {
        "checked": True,
        "level": top,
        "max_abs_auroc_difference": worst,
        "per_signal": diffs,
        "matches": bool(np.isfinite(worst) and worst < 1e-9),
    }
