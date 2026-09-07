"""Round 11 (A3): the fixed-label relabel is committed as its own results file,
and the labeler-variance measurement it produced stands.

Run #2b was re-labelled under the fixed no-judge rule (`data/run2b/labels_fixed.parquet`);
`unc-bench analyze` then produced a schema-identical `results_run2b_fixedlabels.json`
(n=119 — the single rule-ambiguous row is excluded). This file pins the parts
that decide the round's claims: the corrected class counts, that the headline
length-correlated signals lose ground once the length-biased labels are fixed,
and that the variance measurement validated its estimator against the
committed results file before use. These are regression guards: if a future
relabel or analysis edit moves the measurement, they fail here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _fixed() -> Any:
    return json.loads((REPO / "results_run2b_fixedlabels.json").read_text(encoding="utf-8"))


def _old() -> Any:
    return json.loads((REPO / "results_run2b.json").read_text(encoding="utf-8"))


def test_fixed_results_are_the_relabelled_run() -> None:
    results = _fixed()
    assert results["run_name"] == "run2b_fixedlabels"
    view = results["views"]["primary"]
    assert (view["n"], view["n_incorrect"], view["n_correct"]) == (119, 68, 51)
    # the rule-unresolved row is excluded and counted, not coerced
    assert results["labels"]["n_ambiguous_dropped"] == 1
    # gates still fail on the same three grounds (no humans have labelled)
    assert results["validity_gates"]["failed"] == [
        "per_dataset_class_counts",
        "labeling_protocol_validated",
        "human_label_coverage",
    ]


def test_variance_estimator_validated_against_the_committed_file() -> None:
    variance = json.loads(
        (REPO / "data" / "labeler_variance_run2b.json").read_text(encoding="utf-8")
    )
    assert variance["validation"]["estimator_reproduces_results_run2b_json"] is True
    assert variance["unresolved_rows"] == ["triviaqa-jp_1520"]


def test_length_correlated_signals_lose_ground_on_fixed_labels() -> None:
    """The round's central measurement: correcting the length-biased labels
    attenuates the length-correlated signals. Each assertion is a signed
    inequality on committed values, not a number that could drift cosmetically."""
    variance = json.loads(
        (REPO / "data" / "labeler_variance_run2b.json").read_text(encoding="utf-8")
    )
    per = variance["per_signal"]
    # like-for-like delta on the shared 119 rows, negative and past its noise
    assert per["a_total_logprob"]["delta_shared_119"]["point"] < -0.05
    assert per["t_answer_length"]["delta_shared_119"]["point"] < -0.05
    assert per["a_length_normalized_logprob"]["delta_shared_119"]["point"] < -0.05
    # stratified pooled AUROC (the README ranking's sort key) drops for the
    # length baselines when the label bias is removed
    old = _old()["views"]["primary"]["stratified"]["signals"]
    fixed = _fixed()["views"]["primary"]["stratified"]["signals"]
    for name in ("a_total_logprob", "t_answer_length", "a_length_normalized_logprob"):
        assert fixed[name]["point"] < old[name]["point"], name
    # the fixed-label PopQA leader is no longer a family-A total-logprob figure
    assert (
        fixed["a_total_logprob"]["point"] < fixed["b_mean_pairwise_f1"]["point"]
        or fixed["a_total_logprob"]["point"] < fixed["c_p_true_plain"]["point"]
    )
