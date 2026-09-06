"""Decompose run #2b's E4 move into rule effect vs dataset effect (Part P1.1).

Run #2b changed two things at once versus run #2: the dataset was
decontaminated AND the labeling rule changed (heuristic fallback, no judges).
From the SAME committed run #2b generations, this builds three label sets:

  L_exact  : normalized exact match only; misses are unlabeled (excluded).
  L_fuzzy  : the shipped rule — exact match, else fuzzy containment either
             way within a 2-token length gap (current run #2b labels).
  L_strict : exact match, plus containment in the shortening direction only
             (the model dropped qualifiers: "Clinton" for "Bill Clinton").
             The length-generous direction (the model adds words: "Bram
             Stoker" for "Stoker") is disabled, with the gap cap kept.

and recomputes per-dataset AUROC for every scored signal under each rule.
The rule effect on fixed rows is AUROC(L_fuzzy) minus AUROC(L_strict); the E4 move
(run #2 judged 0.514 → run #2b fuzzy 0.740 on PopQA mean-logprob) minus the
rule effect is what decontamination can claim, modulo the judge-vs-heuristic
difference, which no committed artifact can separate and which is stated, not
adjusted away.

Also measures the labeler's own length bias (P1.2): Spearman correlation
between answer token length and P(label = correct) under L_fuzzy vs L_exact,
with a bootstrap interval on the difference.

Reads run #2b artifacts only; writes data/label_rule_sensitivity.json. Labels
nothing by hand and invents nothing:

    uv run python scripts/decompose_run2b_delta.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "data" / "run2b"
REPORT_PATH = REPO_ROOT / "data" / "label_rule_sensitivity.json"


def _strict_correct(answer: str, gold_answers: list[str]) -> bool:
    """Exact match or shortening-direction containment only.

    Same normalization and gap cap as the shipped fuzzy rule; the generous
    direction (gold contained in a longer prediction) is the branch under
    suspicion for length bias and is disabled here.
    """
    from unc_bench.labeling import _contains
    from unc_bench.normalize import exact_match, normalize_answer

    if exact_match(answer, gold_answers):
        return True
    predicted = normalize_answer(answer)
    if not predicted:
        return False
    pred_tokens = predicted.split()
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens or len(pred_tokens) > len(gold_tokens):
            continue
        if abs(len(pred_tokens) - len(gold_tokens)) > 2:
            continue
        if _contains(gold_tokens, pred_tokens):
            return True
    return False


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    import pandas as pd

    from unc_bench.analysis.metrics import auroc
    from unc_bench.normalize import clean_model_answer, exact_match, tokenize

    generations = pd.read_parquet(ARTIFACTS / "generations.parquet")
    actc = pd.read_parquet(ARTIFACTS / "signals_actc.parquet")
    family_b = pd.read_parquet(ARTIFACTS / "signals_b.parquet")
    shipped = pd.read_parquet(ARTIFACTS / "labels.parquet")

    signals = actc.merge(family_b, on="qid", how="left", validate="one_to_one")
    frame = signals.merge(generations, on="qid", how="inner", validate="one_to_one")
    shipped_map = dict(zip(shipped["qid"].astype(str), shipped["label"].astype(str), strict=True))

    import json as _json

    def aliases_of(raw: object) -> list[str]:
        try:
            decoded = _json.loads(str(raw))
        except (TypeError, ValueError):
            return []
        return [str(a) for a in decoded] if isinstance(decoded, list) else []

    exact_labels: dict[str, bool | None] = {}
    strict_incorrect: dict[str, bool] = {}
    for record in frame.to_dict(orient="records"):
        qid = str(record["qid"])
        gold = aliases_of(record.get("gold_answers"))
        answer = clean_model_answer(str(record.get("greedy_text") or ""))
        if exact_match(answer, gold):
            exact_labels[qid] = True
            strict_incorrect[qid] = False
        else:
            exact_labels[qid] = None
            # NOTE: True means INCORRECT here, matching the shipped mapping
            # below (auroc's positive class). An earlier version stored
            # correctness and reported a sign-flipped 0.260; caught because
            # 1 - 0.740 == 0.260 exactly. See AUDIT_RESPONSE round 5.
            strict_incorrect[qid] = not _strict_correct(answer, gold)

    names = [c for c in signals.columns if c != "qid"]
    datasets = frame.set_index(frame["qid"].astype(str))["dataset"].astype(str).to_dict()
    table: dict[str, Any] = {}
    for name in names:
        scores = frame.set_index(frame["qid"].astype(str))[name].to_numpy(dtype=np.float64)
        qids = frame["qid"].astype(str).tolist()
        entry: dict[str, Any] = {}
        for rule, mapping in (
            ("L_fuzzy", {q: shipped_map[q] == "incorrect" for q in qids}),
            ("L_strict", {q: strict_incorrect[q] for q in qids}),
        ):
            y = np.array([mapping[q] for q in qids], dtype=bool)
            entry[rule] = {
                "auroc_pooled": auroc(scores, y),
                "n": int(len(y)),
            }
            for dataset in ("popqa", "triviaqa"):
                mask = np.array([datasets[q] == dataset for q in qids])
                entry[rule][dataset] = auroc(scores[mask], y[mask])
        exact_qids = [q for q in qids if exact_labels[q] is not None]
        exact_idx = np.array([q in set(exact_qids) for q in qids])
        y_exact = np.array([exact_labels[q] for q in exact_qids], dtype=bool)
        entry["L_exact"] = {
            "auroc_pooled": auroc(scores[exact_idx], y_exact),
            "n": int(len(exact_qids)),
            "note": "non-matching rows excluded, so n differs; not directly comparable",
        }
        for dataset in ("popqa", "triviaqa"):
            sub = [q for q in exact_qids if datasets[q] == dataset]
            idx = np.array([q in set(sub) for q in qids])
            entry["L_exact"][dataset] = auroc(scores[idx], np.array([exact_labels[q] for q in sub]))
        table[name] = entry

    # Rule effect with an interval, on the headline signal and dataset.
    popqa = np.array([datasets[q] == "popqa" for q in frame["qid"].astype(str).tolist()])
    qids_all = frame["qid"].astype(str).tolist()
    mean_lp = frame.set_index(frame["qid"].astype(str))["a_mean_logprob"].to_numpy(dtype=np.float64)
    y_fuzzy = np.array([shipped_map[q] == "incorrect" for q in qids_all])[popqa]
    y_strict = np.array([strict_incorrect[q] for q in qids_all])[popqa]
    s_pop = mean_lp[popqa]
    rule_effect = auroc(s_pop, y_fuzzy) - auroc(s_pop, y_strict)
    rng = np.random.default_rng(31337)
    diffs = []
    pos = np.flatnonzero(y_fuzzy)
    neg = np.flatnonzero(~y_fuzzy)
    for _ in range(2000):
        idx = np.concatenate(
            [
                pos[rng.integers(0, pos.size, size=pos.size)],
                neg[rng.integers(0, neg.size, size=neg.size)],
            ]
        )
        diffs.append(auroc(s_pop[idx], y_fuzzy[idx]) - auroc(s_pop[idx], y_strict[idx]))
    # NOTE: resampling is stratified on the FUZZY labels; the strict labels
    # ride along. A resample degenerate under strict labels yields NaN and is
    # dropped, which is reported rather than hidden.
    finite = [d for d in diffs if d == d]
    rule_effect_ci = [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    # P1.2: labeler length bias under each rule (PopQA rows).
    answer_len = np.array(
        [len(tokenize(clean_model_answer(str(t) or ""))) for t in frame["greedy_answer"].tolist()]
    )[popqa]
    correct_fuzzy = (~y_fuzzy).astype(float)
    correct_strict = (~y_strict).astype(float)
    from unc_bench.analysis.metrics import _spearman

    rho_fuzzy = _spearman(answer_len.astype(float), correct_fuzzy)
    rho_strict = _spearman(answer_len.astype(float), correct_strict)
    # L_exact on the PopQA subset only, over its own rows.
    exact_pop = [q for q in qids_all if datasets[q] == "popqa" and exact_labels[q] is not None]
    exact_len = np.array(
        [
            len(
                tokenize(
                    clean_model_answer(
                        str(frame.set_index(frame["qid"].astype(str)).loc[q, "greedy_answer"]) or ""
                    )
                )
            )
            for q in exact_pop
        ],
        dtype=float,
    )
    exact_correct = np.array([not exact_labels[q] for q in exact_pop], dtype=float)
    rho_exact = _spearman(exact_len, exact_correct)

    report = {
        "rows": len(frame),
        "rules": {
            "L_exact": "normalized exact match only; misses excluded",
            "L_fuzzy": "shipped run #2b labels (exact + either-way containment, gap 2)",
            "L_strict": "exact + shortening-direction containment only (gap cap kept)",
        },
        "per_signal": table,
        "e4_decomposition_a_mean_logprob_popqa": {
            "run2_judged": 0.514,
            "run2b_L_fuzzy": table["a_mean_logprob"]["L_fuzzy"]["popqa"],
            "run2b_L_strict": table["a_mean_logprob"]["L_strict"]["popqa"],
            "rule_effect_fuzzy_minus_strict": rule_effect,
            "rule_effect_95ci": rule_effect_ci,
            "n_dropped_nonfinite": len(diffs) - len(finite),
        },
        "length_bias_popqa": {
            "spearman_length_vs_correct_L_fuzzy": rho_fuzzy,
            "spearman_length_vs_correct_L_strict": rho_strict,
            "spearman_length_vs_correct_L_exact": rho_exact,
            "L_exact_note": (
                "L_exact rows are all correct by construction (misses excluded), "
                "so the correlation is undefined (NaN); L_strict is the "
                "informative contrast on identical rows"
            ),
            "n_exact_rows": len(exact_pop),
        },
    }

    def _clean(value: object) -> object:
        if isinstance(value, float) and value != value:  # NaN
            return None
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    REPORT_PATH.write_text(
        json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(f"wrote {REPORT_PATH}", flush=True)
    e4 = report["e4_decomposition_a_mean_logprob_popqa"]
    print(
        f"E4: fuzzy={e4['run2b_L_fuzzy']:.3f} strict={e4['run2b_L_strict']:.3f} "
        f"rule_effect={rule_effect:.3f} {rule_effect_ci}",
        flush=True,
    )
    print(f"length bias: fuzzy rho={rho_fuzzy:.3f} exact rho={rho_exact:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
