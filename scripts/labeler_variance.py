"""Labeler-induced variance on run #2b: every AUROC under the old vs the fixed
label set, with a like-for-like delta on the rows both sets can score (A3).

`results_run2b.json` was produced from the pre-fix heuristic labels (n=120);
`results_run2b_fixedlabels.json` from the fixed no-judge rule (n=119 — the one
rule-ambiguous row is excluded pending a human). Both were produced by the same
`unc-bench analyze` pipeline, so the two files are already comparable. But the
row sets differ by one row, so the honest per-signal delta needs both label
sets scored on the SAME rows. This script therefore:

  1. VALIDATES its estimator: recomputes the pooled and per-dataset AUROC CI
     for every signal under the committed labels on all 120 rows and asserts
     it matches `results_run2b.json` to machine precision. A mismatch means
     the recomputation is not the pipeline's and nothing here can be trusted.
  2. Computes the fixed-label AUROC CI for every signal on the shared 119 rows,
     the old-label AUROC CI on those same 119 rows, and a paired stratified
     bootstrap CI on the difference (one shared draw sequence per signal).

Writes `data/labeler_variance_run2b.json`. Reads only committed artifacts and
the two results files; never touches `human_label`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The one run #2b row the fixed rule cannot decide (partial-alias echo); it is
#: excluded from both label sets in the like-for-like comparison.
UNRESOLVED_QID = "triviaqa-jp_1520"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    import pandas as pd

    from unc_bench.analysis.metrics import (
        _stratified_cluster_draw,
        auroc,
        bootstrap_auroc_ci,
        usable_mask,
    )
    from unc_bench.config import Config
    from unc_bench.normalize import normalize_answer
    from unc_bench.stages.common import json_load, read_checkpoint

    cfg = Config.load("configs/run2b_clean.yaml")
    resamples = cfg.analysis.bootstrap_resamples
    seed = cfg.analysis.bootstrap_seed
    level = cfg.analysis.ci_level

    old_results = json.loads((REPO_ROOT / "results_run2b.json").read_text(encoding="utf-8"))
    new_results = json.loads(
        (REPO_ROOT / "results_run2b_fixedlabels.json").read_text(encoding="utf-8")
    )

    # ---- reconstruct the analysis frame exactly as report.py does -----------
    run_dir = REPO_ROOT / "data" / "run2b"
    from unc_bench.signals.base import signal_names
    from unc_bench.stages.score_signals import merged_signals

    signals = merged_signals(cfg)
    labels_old = read_checkpoint(run_dir / "labels.parquet")
    labels_new = read_checkpoint(run_dir / "labels_fixed.parquet")
    dataset_frame = read_checkpoint(run_dir / "dataset.parquet")
    generations = read_checkpoint(run_dir / "generations.parquet")
    merged = signals.merge(labels_old, on="qid", how="inner", validate="one_to_one")
    merged = merged.merge(
        dataset_frame[["qid", "dataset", "question"]], on="qid", validate="one_to_one"
    )
    if generations is not None and "greedy_answer" in generations.columns:
        merged = merged.merge(
            generations[["qid", "greedy_answer"]], on="qid", validate="one_to_one"
        )
    merged = merged.reset_index(drop=True)

    labels_by_qid_new = dict(zip(labels_new["qid"].astype(str), labels_new["label"].astype(str)))
    names = [n for n in signal_names() if n in signals.columns]
    qids = merged["qid"].astype(str).tolist()

    # One ambiguous row under the fixed rule: excluded from the like-for-like
    # set (and from the fixed analysis set). Build both y vectors.
    y_old_all = (merged["label"] == "incorrect").to_numpy(dtype=bool)
    keep_all = np.ones(len(merged), dtype=bool)
    keep_shared = np.array([q != UNRESOLVED_QID for q in qids], dtype=bool)
    y_new_all = np.array([labels_by_qid_new[q] == "incorrect" for q in qids], dtype=bool)

    # cluster ids on the FULL row set (question-text clusters); sliced below.
    question_codes = (
        merged["question"]
        .astype(str)
        .map(normalize_answer)
        .astype("category")
        .cat.codes.to_numpy(dtype=np.int64)
    )

    def block(old_y: np.ndarray, row_mask: np.ndarray) -> dict[str, Any]:
        """Per-signal AUROC CI (pooled) over the masked rows, using the masked
        label vector, reproducing report.py's estimator."""
        out: dict[str, Any] = {}
        idx = np.flatnonzero(row_mask)
        clusters = question_codes[idx] if cfg.analysis.cluster_bootstrap else None
        for name in names:
            scores = merged[name].to_numpy(dtype=np.float64)[idx]
            y = old_y[idx]
            ci = bootstrap_auroc_ci(
                scores,
                y,
                resamples=resamples,
                seed=seed,
                level=level,
                cluster_ids=clusters,
            )
            out[name] = ci.as_dict()
        return out

    def per_dataset(
        old_y: np.ndarray, row_mask: np.ndarray
    ) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = {}
        idx_full = np.flatnonzero(row_mask)
        for source in ("popqa", "triviaqa"):
            src_mask = row_mask & (merged["dataset"] == source).to_numpy(dtype=bool)
            idx = np.flatnonzero(src_mask)
            clusters = question_codes[idx] if cfg.analysis.cluster_bootstrap else None
            out[source] = {}
            for name in names:
                scores = merged[name].to_numpy(dtype=np.float64)[idx]
                y = old_y[idx]
                ci = bootstrap_auroc_ci(
                    scores,
                    y,
                    resamples=resamples,
                    seed=seed,
                    level=level,
                    cluster_ids=clusters,
                )
                out[source][name] = ci.as_dict()
        return out

    # ---- 1. estimator validation against the committed results file ---------
    old_all = block(y_old_all, keep_all)
    old_all_pd = per_dataset(y_old_all, keep_all)
    max_abs = 0.0
    for name in names:
        got = old_all[name]
        want = old_results["views"]["primary"]["signals"][name]["auroc"]
        for key in ("point", "ci_low", "ci_high"):
            max_abs = max(max_abs, abs(float(got[key]) - float(want[key])))
        for source in ("popqa", "triviaqa"):
            w = old_results["views"]["primary"]["per_dataset"]["datasets"][source]["signals"][name]
            g = old_all_pd[source][name]
            for key in ("point", "ci_low", "ci_high"):
                max_abs = max(max_abs, abs(float(g[key]) - float(w["auroc_ci"][key])))
    assert (
        max_abs < 1e-9
    ), f"estimator does not reproduce results_run2b.json: max abs diff {max_abs}"

    # ---- 2. like-for-like fixed vs old on the shared 119 rows ---------------
    old_shared = block(y_old_all, keep_shared)
    new_shared = block(y_new_all, keep_shared)
    new_all = block(y_new_all, keep_all)
    old_shared_pd = per_dataset(y_old_all, keep_shared)
    new_shared_pd = per_dataset(y_new_all, keep_shared)

    # paired delta CI: one shared stratified draw sequence per signal
    def delta_ci(
        name: str, y_old: np.ndarray, y_new: np.ndarray, row_mask: np.ndarray
    ) -> dict[str, Any]:
        idx = np.flatnonzero(row_mask)
        s = merged[name].to_numpy(dtype=np.float64)[idx]
        yo, yn = y_old[idx], y_new[idx]
        keep = usable_mask(s)
        s, yo, yn = s[keep], yo[keep], yn[keep]
        point = auroc(s, yn) - auroc(s, yo)
        pos = np.flatnonzero(yn)
        neg = np.flatnonzero(~yn)
        clusters = question_codes[idx][keep] if cfg.analysis.cluster_bootstrap else None
        if clusters is not None and clusters.shape != yn.shape:
            clusters = None
        rng = np.random.default_rng(seed)
        draws = np.empty(resamples, dtype=np.float64)
        for r in range(resamples):
            if clusters is not None:
                take = _stratified_cluster_draw(yn, clusters, rng)
            else:
                tp = rng.integers(0, pos.size, size=pos.size)
                tn = rng.integers(0, neg.size, size=neg.size)
                take = np.concatenate([pos[tp], neg[tn]])
            draws[r] = auroc(s[take], yn[take]) - auroc(s[take], yo[take])
        finite = draws[np.isfinite(draws)]
        alpha = (1.0 - level) / 2.0
        return {
            "point": point,
            "ci_low": float(np.quantile(finite, alpha)),
            "ci_high": float(np.quantile(finite, 1.0 - alpha)),
            "n": int(keep.sum()),
            "resamples": int(finite.size),
        }

    per_signal: dict[str, Any] = {}
    for name in names:
        per_signal[name] = {
            "old_pooled_120": old_all[name],
            "new_pooled_119": new_all[name],
            "old_pooled_shared_119": old_shared[name],
            "new_pooled_shared_119": new_shared[name],
            "delta_shared_119": delta_ci(name, y_old_all, y_new_all, keep_shared),
            "per_dataset_old_120": {src: old_all_pd[src][name] for src in ("popqa", "triviaqa")},
            "per_dataset_old_shared_119": {
                src: old_shared_pd[src][name] for src in ("popqa", "triviaqa")
            },
            "per_dataset_new_shared_119": {
                src: new_shared_pd[src][name] for src in ("popqa", "triviaqa")
            },
        }

    def class_counts(y: np.ndarray, row_mask: np.ndarray) -> dict[str, int]:
        return {
            "incorrect": int(np.count_nonzero(y[row_mask])),
            "correct": int(row_mask.sum() - np.count_nonzero(y[row_mask])),
        }

    payload: dict[str, Any] = {
        "run": "run2b_clean",
        "note": (
            "AUROC for INCORRECT under the pre-fix heuristic labels and the "
            "fixed no-judge rule, pooled and per dataset. 120 = the committed "
            "analysis set (results_run2b.json); 119 = the rows both label sets "
            "can score, excluding the single rule-ambiguous row "
            f"({UNRESOLVED_QID}, queued for a human). delta_shared_119 is a "
            "paired stratified bootstrap on the difference; estimator validated "
            "by exact reproduction of results_run2b.json before use."
        ),
        "n": {
            "old": int(keep_all.sum()),
            "fixed": int(keep_shared.sum()),
            "shared": int(keep_shared.sum()),
        },
        "class_counts": {
            "old_120": class_counts(y_old_all, keep_all),
            "old_shared_119": class_counts(y_old_all, keep_shared),
            "fixed_shared_119": class_counts(y_new_all, keep_shared),
        },
        "unresolved_rows": [UNRESOLVED_QID],
        "changed_rows": json.loads((run_dir / "label_fix_rows.json").read_text(encoding="utf-8"))[
            "changes"
        ],
        "per_signal": per_signal,
        "validation": {
            "estimator_reproduces_results_run2b_json": True,
            "max_abs_diff_over_all_points_and_ci_bounds": max_abs,
        },
    }
    out_path = REPO_ROOT / "data" / "labeler_variance_run2b.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    # print a compact table
    print("per-signal pooled AUROC (old-120 | old-119 | new-119 | delta [CI]):")
    for name in sorted(per_signal, key=lambda n: -per_signal[n]["new_pooled_shared_119"]["point"]):
        o = per_signal[name]["old_pooled_120"]["point"]
        os = per_signal[name]["old_pooled_shared_119"]["point"]
        ns = per_signal[name]["new_pooled_shared_119"]["point"]
        d = per_signal[name]["delta_shared_119"]
        print(
            f"  {name:34s} {o:6.3f} {os:6.3f} {ns:6.3f}  {d['point']:+.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
