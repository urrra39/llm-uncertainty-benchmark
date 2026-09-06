"""The fuzzy labeler must not buy AUROC with answer length (Part P1.2).

Run #2b changed the labeling rule alongside the dataset, so the E4 move
could have been a length-mediated labeling artifact: fuzzy containment
accepts longer answers more readily, and logprob signals correlate with
length. The committed decomposition (`data/label_rule_sensitivity.json`,
recomputed here from the committed run #2b artifacts) shows the generous
containment branch fired zero times in 120 rows and the fuzzy-vs-strict rule
effect is exactly 0.000 — these tests pin that, so a future edit to the
labeler that reintroduces length generosity fails here rather than silently
moving E4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUN2B = REPO / "data" / "run2b"


def _frame() -> pd.DataFrame:
    generations = pd.read_parquet(RUN2B / "generations.parquet")
    labels = pd.read_parquet(RUN2B / "labels.parquet")
    return generations.merge(labels, on="qid", how="inner", validate="one_to_one")


def test_generous_containment_fired_zero_times() -> None:
    """Only exact matches and shortening-direction containment label rows
    correct; the length-generous direction (gold inside a longer prediction)
    decided nothing in this run."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from unc_bench.labeling import _contains
    from unc_bench.normalize import clean_model_answer, exact_match, normalize_answer

    def aliases_of(raw: object) -> list[str]:
        try:
            decoded = json.loads(str(raw))
        except (TypeError, ValueError):
            return []
        return [str(a) for a in decoded] if isinstance(decoded, list) else []

    generous = 0
    for record in _frame().to_dict(orient="records"):
        gold = aliases_of(record.get("gold_answers"))
        answer = clean_model_answer(str(record.get("greedy_text") or ""))
        if exact_match(answer, gold):
            continue
        pred = normalize_answer(answer).split()
        for alias in gold:
            gold_tokens = normalize_answer(alias).split()
            if (
                len(pred) > len(gold_tokens)
                and abs(len(pred) - len(gold_tokens)) <= 2
                and _contains(pred, gold_tokens)
            ):
                generous += 1
                break
    assert generous == 0


def test_decomposition_pins_zero_rule_effect() -> None:
    report = json.loads((REPO / "data" / "label_rule_sensitivity.json").read_text())
    e4 = report["e4_decomposition_a_mean_logprob_popqa"]
    assert e4["rule_effect_fuzzy_minus_strict"] == 0.0
    assert e4["rule_effect_95ci"] == [0.0, 0.0]
    assert e4["run2b_L_fuzzy"] == e4["run2b_L_strict"]


def test_length_bias_is_identical_under_both_rules() -> None:
    report = json.loads((REPO / "data" / "label_rule_sensitivity.json").read_text())
    bias = report["length_bias_popqa"]
    assert bias["spearman_length_vs_correct_L_fuzzy"] == bias["spearman_length_vs_correct_L_strict"]
    # L_exact rows are all correct by construction, so its correlation is
    # undefined rather than zero — pinned as null, not as a number.
    assert bias["spearman_length_vs_correct_L_exact"] is None
