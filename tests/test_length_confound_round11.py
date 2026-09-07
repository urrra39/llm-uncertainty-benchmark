"""Round 11 (B1): the length confound is measured per signal, and its shape is
pinned.

The mechanism the round demonstrates — a length-biased labeler plus a
length-correlated signal manufactures AUROC — makes four signed, committed
claims that this file pins as inequalities on `data/length_confound_audit.json`:

  - total log probability is more correlated with answer length than with the
    label, and its partial association with the label (length removed) is
    smaller than its raw one;
  - the token-count baseline `t_answer_length` is answer length by definition,
    so removing length from it leaves almost nothing (partial ~ 0);
  - verification confidence `c_p_true_plain` is essentially length-independent,
    which is why it is the one top signal that does NOT lose ground under the
    fixed labels;
  - under the fixed labels, `a_total_logprob`'s within-long-answer-stratum
    AUROC collapses toward its short-stratum value — the length bias is where
    its old skill lived.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _audit() -> dict:
    return json.loads((REPO / "data" / "length_confound_audit.json").read_text(encoding="utf-8"))


def test_total_logprob_is_more_entangled_with_length_than_label() -> None:
    old = _audit()["old"]["a_total_logprob"]
    assert old["spearman_with_length"] > old["spearman_with_label"]
    assert old["partial_spearman_given_length"] < old["spearman_with_label"]


def test_length_baseline_has_nothing_left_once_length_is_removed() -> None:
    old = _audit()["old"]["t_answer_length"]
    assert old["spearman_with_length"] == 1.0  # it IS the length variable
    assert old["partial_spearman_given_length"] < 0.2


def test_verification_confidence_is_length_independent() -> None:
    old = _audit()["old"]["c_p_true_plain"]
    assert old["spearman_with_length"] < 0.1
    # and it is the only headline signal whose fixed-label pooled AUROC rises
    fixed = _audit()["fixed"]["c_p_true_plain"]["pooled_auroc"]["point"]
    assert fixed > old["pooled_auroc"]["point"]


def test_total_logprob_long_stratum_skill_was_length_bias() -> None:
    old = _audit()["old"]["a_total_logprob"]
    fixed = _audit()["fixed"]["a_total_logprob"]
    old_gap = (
        old["strata"]["above_median"]["auroc"]["point"]
        - old["strata"]["at_or_below_median"]["auroc"]["point"]
    )
    fixed_gap = (
        fixed["strata"]["above_median"]["auroc"]["point"]
        - fixed["strata"]["at_or_below_median"]["auroc"]["point"]
    )
    assert old_gap > 0.1  # old labels: long answers looked far more separable
    assert fixed_gap < 0.05  # fixed labels: the long-stratum edge is gone
