"""Length confound, quantified per signal (B1, round 11).

The pre-fix heuristic labeler is length-biased (fuzzy containment rejected
verbatim-correct long answers; Spearman(answer length, correct) = -0.26 on
PopQA, data/label_rule_sensitivity.json), and the top of the table is family A
total log probability, which is mechanically more negative for longer answers.
So part of every AUROC against the old labels may be the labeler paying signals
for tracking answer length. A3 measured the labeler-induced variance directly;
this script quantifies the remaining overlap per signal on the same rows:

  1. Spearman(signal, label) and Spearman(signal, answer length) — how each
     signal is entangled with the label and with length.
  2. Partial Spearman of signal with label after residualizing on answer
     length (rank residualization), for every signal — the association that
     survives once length is removed from both sides.
  3. AUROC with bootstrap CI within answer-length strata (at/below vs above the
     median) for every signal — does the signal separate correct from
     incorrect among answers of comparable length?

Both the pre-fix labels (results_run2b.json's) and the fixed labels
(results_run2b_fixedlabels.json's) are scored on the SAME 119 rows that both
label sets can decide, so the two columns are directly comparable; the one
rule-ambiguous row is excluded. The labeler-variance script validated that
this estimator reproduces results_run2b.json exactly before use.

Writes data/length_confound_audit.json. Reads only committed artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UNRESOLVED_QID = "triviaqa-jp_1520"


def _ols_residual(x: Any, z: Any) -> Any:
    """Residuals of x on z (both numeric arrays) via least squares."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    z1 = np.column_stack([np.ones(z.size), z])
    beta, *_ = np.linalg.lstsq(z1, x, rcond=None)
    return x - z1 @ beta


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    import pandas as pd

    from unc_bench.analysis.metrics import _midranks, _spearman, bootstrap_auroc_ci, usable_mask
    from unc_bench.config import Config
    from unc_bench.normalize import normalize_answer
    from unc_bench.stages.common import read_checkpoint
    from unc_bench.stages.score_signals import merged_signals

    cfg = Config.load("configs/run2b_clean.yaml")
    run_dir = REPO_ROOT / "data" / "run2b"
    signals = merged_signals(cfg)
    labels_old = read_checkpoint(run_dir / "labels.parquet")
    labels_new = read_checkpoint(run_dir / "labels_fixed.parquet")
    dataset_frame = read_checkpoint(run_dir / "dataset.parquet")
    merged = signals.merge(labels_old, on="qid", how="inner", validate="one_to_one")
    merged = merged.merge(dataset_frame[["qid", "dataset", "question"]], on="qid", validate="one_to_one")
    merged = merged.reset_index(drop=True)

    from unc_bench.signals.base import signal_names

    names = [n for n in signal_names() if n in signals.columns]
    qids = merged["qid"].astype(str).tolist()
    shared = np.array([q != UNRESOLVED_QID for q in qids], dtype=bool)
    idx = np.flatnonzero(shared)

    labels_by_qid_new = dict(zip(labels_new["qid"].astype(str), labels_new["label"].astype(str)))
    y_old = (merged["label"] == "incorrect").to_numpy(dtype=bool)[idx]
    y_new = np.array([labels_by_qid_new[q] == "incorrect" for q in qids], dtype=bool)[idx]

    question_codes = (
        merged["question"].astype(str).map(normalize_answer).astype("category").cat.codes.to_numpy(dtype=np.int64)[idx]
    )
    length_full = merged["t_answer_length"].to_numpy(dtype=np.float64)[idx]

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        return float(_spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))

    def partial_spearman(sig: np.ndarray, y: np.ndarray, length: np.ndarray) -> float:
        keep = usable_mask(sig) & usable_mask(length)
        if int(np.count_nonzero(keep)) < 5:
            return float("nan")
        r_sig = _midranks(sig[keep])
        r_y = _midranks(y[keep])
        r_len = _midranks(length[keep])
        e_sig = _ols_residual(r_sig, r_len)
        e_y = _ols_residual(r_y, r_len)
        return float(_spearman(e_sig, e_y))

    def per_signal_block(y: np.ndarray, row_mask: np.ndarray) -> dict[str, Any]:
        rows = np.flatnonzero(row_mask)
        clusters = question_codes[rows] if cfg.analysis.cluster_bootstrap else None
        out: dict[str, Any] = {}
        length = length_full[rows]
        usable_len = usable_mask(length)
        med = float(np.median(length[usable_len]))
        for name in names:
            sig = merged[name].to_numpy(dtype=np.float64)[idx][rows]
            keep = usable_mask(sig) & usable_len
            if int(np.count_nonzero(keep)) < 5:
                out[name] = {"note": "too few usable rows to estimate"}
                continue
            low = keep & (length <= med)
            high = keep & (length > med)
            block: dict[str, Any] = {
                "spearman_with_label": spearman(sig[keep], y[rows][keep]),
                "spearman_with_length": spearman(sig[keep], length[keep]),
                "partial_spearman_given_length": partial_spearman(sig, y[rows], length),
                "median_answer_length": med,
                "strata": {},
                "pooled_auroc": bootstrap_auroc_ci(
                    sig, y[rows], resamples=cfg.analysis.bootstrap_resamples,
                    seed=cfg.analysis.bootstrap_seed, level=cfg.analysis.ci_level,
                    cluster_ids=clusters,
                ).as_dict(),
            }
            for label, mask in (("at_or_below_median", low), ("above_median", high)):
                s = sig[mask]
                ys = y[rows][mask]
                clusters_s = clusters[mask] if clusters is not None else None
                block["strata"][label] = {
                    "n": int(np.count_nonzero(mask)),
                    "n_incorrect": int(np.count_nonzero(ys)),
                    "auroc": bootstrap_auroc_ci(
                        s, ys, resamples=cfg.analysis.bootstrap_resamples,
                        seed=cfg.analysis.bootstrap_seed, level=cfg.analysis.ci_level,
                        cluster_ids=clusters_s,
                    ).as_dict(),
                }
            out[name] = block
        return out

    old_block = per_signal_block(y_old, np.ones(idx.size, dtype=bool))
    fixed_block = per_signal_block(y_new, np.ones(idx.size, dtype=bool))

    payload: dict[str, Any] = {
        "run": "run2b_clean",
        "n_shared_rows": int(idx.size),
        "unresolved_excluded": UNRESOLVED_QID,
        "note": (
            "every signal scored on the same 119 rows both label sets decide, "
            "under the pre-fix labels (old) and the fixed labels (fixed); strata "
            "split at the median normalized answer length; partial Spearman = "
            "rank-residualized association of signal with label given length"
        ),
        "length_definition": "t_answer_length (normalized answer token count)",
        "old": old_block,
        "fixed": fixed_block,
    }
    out_path = REPO_ROOT / "data" / "length_confound_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    print("signal           r(label) r(length) partial | pooled old/fixed")
    for name in sorted(names):
        o, f = old_block.get(name), fixed_block.get(name)
        if o is None or "spearman_with_label" not in o:
            continue
        print(
            f"{name:30s} {o['spearman_with_label']:7.3f} {o['spearman_with_length']:7.3f} "
            f"{o['partial_spearman_given_length']:7.3f} | "
            f"{o['pooled_auroc']['point']:.3f}/{f['pooled_auroc']['point']:.3f}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
