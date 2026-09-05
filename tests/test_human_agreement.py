"""Tests for judge-versus-human agreement and the validation CSV's schema.

The utility is never run against real human labels anywhere in this project —
there are none — so these tests supply synthetic label columns to exercise it,
and separately assert that the shipped CSV is in the state the documentation
claims: complete in every column a labeller needs, and empty in `human_label`.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from unc_bench.analysis.human_agreement import (
    HUMAN_COLUMN,
    HUMAN_LABELS,
    MACHINE_COLUMNS,
    build_report,
    read_validation_csv,
    render_report,
    run,
    score_column,
)

SHIPPED = Path("data/human_validation_sample.csv")

REQUIRED_COLUMNS = (
    "qid",
    "dataset",
    "question",
    "gold_answers",
    "model_answer",
    "heuristic_verdict",
    "judge_primary_verdict",
    "judge_secondary_verdict",
    "machine_label",
    "machine_label_source",
    "human_label",
)


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(qid: str, machine: str, human: str = "") -> dict[str, str]:
    return {
        "qid": qid,
        "heuristic_verdict": machine,
        "judge_primary_verdict": machine,
        "judge_secondary_verdict": "",
        "machine_label": machine,
        HUMAN_COLUMN: human,
    }


# ------------------------------------------------------------ the shipped file


def test_shipped_csv_has_every_required_column() -> None:
    rows = read_validation_csv(SHIPPED)
    assert tuple(rows[0].keys()) == REQUIRED_COLUMNS


def test_shipped_csv_has_one_hundred_rows() -> None:
    assert len(read_validation_csv(SHIPPED)) == 100


def test_shipped_human_label_column_is_entirely_empty() -> None:
    """Filling it would be fabrication. It must stay empty in the repository."""
    rows = read_validation_csv(SHIPPED)
    assert all(row[HUMAN_COLUMN] == "" for row in rows)


def test_shipped_csv_gold_answers_are_not_the_corrupted_form() -> None:
    """The column used to hold a character-split repr truncated to 17 chars."""
    for row in read_validation_csv(SHIPPED):
        gold = row["gold_answers"]
        assert gold
        assert not gold.startswith("[ | ")
        assert '" | ' not in gold[:6]


def test_shipped_csv_every_row_has_the_labeller_essentials() -> None:
    for row in read_validation_csv(SHIPPED):
        assert row["question"].strip()
        assert row["gold_answers"].strip()
        assert row["heuristic_verdict"] in HUMAN_LABELS
        assert row["machine_label"] in HUMAN_LABELS
        assert row["machine_label_source"] in ("exact_match", "judge")


def test_shipped_csv_judge_column_is_filled_exactly_on_judged_rows() -> None:
    """An exact-match row never reached a judge, so its judge cell is empty."""
    for row in read_validation_csv(SHIPPED):
        if row["machine_label_source"] == "judge":
            assert row["judge_primary_verdict"] in HUMAN_LABELS
            assert row["judge_primary_verdict"] == row["machine_label"]
        else:
            assert row["judge_primary_verdict"] == ""


def test_shipped_csv_secondary_judge_column_is_filled_on_judged_rows() -> None:
    """Recovered by re-running the second judge over the rows it can be re-asked
    about: exactly the judge-settled rows, whose model answer this file stores.

    An `exact_match` row saw no judge, so a blank there is correct rather than
    missing. See data/README.md for why 13 of run #2's 66 judged rows are not in
    this file and therefore not recoverable.
    """
    rows = read_validation_csv(SHIPPED)
    judged = [r for r in rows if r["machine_label_source"] == "judge"]
    assert len(judged) == 53
    assert all(r["judge_secondary_verdict"] in ("correct", "incorrect") for r in judged)
    assert all(
        r["judge_secondary_verdict"] == "" for r in rows if r["machine_label_source"] != "judge"
    )


def test_recovered_secondary_verdicts_reproduce_the_stored_kappa_direction() -> None:
    """The recovered verdicts must agree with the primary judge on every row.

    Run #2 recorded one disagreement over 66 rows. It is not among the 53 rows
    this file can reproduce, so the recovered subset is unanimous and its kappa
    is 1.0. Pinned so a future edit cannot silently change the recovered column
    without the discrepancy being noticed.
    """
    rows = read_validation_csv(SHIPPED)
    judged = [r for r in rows if r["machine_label_source"] == "judge"]
    disagreements = [
        r["qid"] for r in judged if r["judge_primary_verdict"] != r["judge_secondary_verdict"]
    ]
    assert disagreements == []


def test_shipped_csv_is_balanced_on_the_machine_label() -> None:
    rows = read_validation_csv(SHIPPED)
    correct = sum(1 for r in rows if r["machine_label"] == "correct")
    assert correct == len(rows) - correct == 50


def test_shipped_csv_qids_are_unique() -> None:
    rows = read_validation_csv(SHIPPED)
    assert len({r["qid"] for r in rows}) == len(rows)


def test_shipped_csv_heuristic_disagrees_with_the_machine_label_somewhere() -> None:
    """If these always agreed the heuristic column would carry no information
    and there would be nothing for a human to adjudicate."""
    rows = read_validation_csv(SHIPPED)
    disagreements = sum(1 for r in rows if r["heuristic_verdict"] != r["machine_label"])
    assert disagreements == 9


def test_report_on_the_shipped_file_finds_no_labels() -> None:
    report = build_report(SHIPPED)
    assert report.n_rows == 100
    assert report.n_labelled == 0
    assert report.n_unlabelled == 100
    assert report.invalid_labels == ()
    assert not report.usable


def test_rendered_report_on_the_shipped_file_refuses_to_invent_a_number() -> None:
    text = render_report(build_report(SHIPPED))
    assert "No human labels present" in text
    assert "fabricated" in text
    assert "data/README.md" in text


def test_run_prints_and_returns(capsys: pytest.CaptureFixture[str]) -> None:
    report = run(SHIPPED)
    assert report.n_labelled == 0
    assert "Human validation" in capsys.readouterr().out


# ---------------------------------------------------------- synthetic agreement


def test_perfect_agreement_scores_kappa_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [
            _row("a", "correct", "correct"),
            _row("b", "incorrect", "incorrect"),
            _row("c", "correct", "correct"),
            _row("d", "incorrect", "incorrect"),
        ],
    )
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert scored.n_compared == 4
    assert scored.n_agree == 4
    assert scored.disagreements == 0
    assert scored.kappa == pytest.approx(1.0)
    assert scored.observed_agreement == pytest.approx(1.0)


def test_one_disagreement_lowers_kappa_below_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [
            _row("a", "correct", "correct"),
            _row("b", "incorrect", "incorrect"),
            _row("c", "correct", "incorrect"),
            _row("d", "incorrect", "incorrect"),
        ],
    )
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert (scored.n_agree, scored.n_compared) == (3, 4)
    assert 0.0 < scored.kappa < 1.0


def test_kappa_is_undefined_when_both_sides_are_unanimous(tmp_path: Path) -> None:
    """Chance agreement is 1.0, so kappa is 0/0. NaN, not zero: two labellers
    who both said 'correct' every time did not disagree completely."""
    path = _write(
        tmp_path / "v.csv",
        [_row("a", "correct", "correct"), _row("b", "correct", "correct")],
    )
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert math.isnan(scored.kappa)
    assert not scored.kappa_available
    assert scored.observed_agreement == pytest.approx(1.0)
    assert "undefined" in render_report(build_report(path))


def test_systematic_disagreement_gives_negative_kappa(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [
            _row("a", "correct", "incorrect"),
            _row("b", "incorrect", "correct"),
            _row("c", "correct", "incorrect"),
            _row("d", "incorrect", "correct"),
        ],
    )
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert scored.n_agree == 0
    assert scored.kappa < 0.0


def test_unlabelled_rows_are_skipped_not_counted_as_disagreements(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [
            _row("a", "correct", "correct"),
            _row("b", "incorrect", ""),
            _row("c", "incorrect", "incorrect"),
        ],
    )
    report = build_report(path)
    assert (report.n_labelled, report.n_unlabelled) == (2, 1)
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert scored.n_compared == 2
    assert scored.n_agree == 2


def test_labels_are_case_and_whitespace_insensitive(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [_row("a", "correct", " Correct "), _row("b", "incorrect", "INCORRECT")],
    )
    report = build_report(path)
    assert report.n_labelled == 2
    scored = score_column(read_validation_csv(path), "machine_label")
    assert scored is not None
    assert scored.n_agree == 2


def test_an_unrecognised_label_is_rejected_by_name(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [_row("a", "correct", "probably"), _row("b", "incorrect", "incorrect")],
    )
    report = build_report(path)
    assert report.invalid_labels == (("a", "probably"),)
    assert not report.usable
    text = render_report(report)
    assert "unrecognised label" in text
    assert "probably" in text


def test_a_row_missing_the_machine_verdict_is_excluded(tmp_path: Path) -> None:
    """An exact-match row has no judge verdict; it must not count against the
    judge's agreement."""
    rows = [_row("a", "correct", "correct"), _row("b", "correct", "correct")]
    rows[1]["judge_primary_verdict"] = ""
    path = _write(tmp_path / "v.csv", rows)
    scored = score_column(read_validation_csv(path), "judge_primary_verdict")
    assert scored is not None
    assert scored.n_compared == 1


def test_an_absent_column_scores_none_rather_than_zero(tmp_path: Path) -> None:
    path = _write(tmp_path / "v.csv", [_row("a", "correct", "correct")])
    assert score_column(read_validation_csv(path), "not_a_column") is None


def test_a_present_but_empty_column_reports_no_comparable_rows(tmp_path: Path) -> None:
    path = _write(tmp_path / "v.csv", [_row("a", "correct", "correct")])
    scored = score_column(read_validation_csv(path), "judge_secondary_verdict")
    assert scored is not None
    assert scored.n_compared == 0
    assert not scored.kappa_available


def test_every_machine_column_is_scored_on_a_filled_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "v.csv",
        [_row("a", "correct", "correct"), _row("b", "incorrect", "incorrect")],
    )
    report = build_report(path)
    assert {a.machine_column for a in report.agreements} == set(MACHINE_COLUMNS)
    assert report.usable


def test_report_states_that_this_is_not_the_judge_versus_judge_kappa(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "v.csv",
        [_row("a", "correct", "correct"), _row("b", "incorrect", "incorrect")],
    )
    text = render_report(build_report(path))
    assert "different" in text
    assert "results.json" in text


def test_a_file_without_the_human_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "v.csv"
    path.write_text("qid,machine_label\na,correct\n", encoding="utf-8")
    with pytest.raises(KeyError, match=HUMAN_COLUMN):
        build_report(path)


def test_an_empty_file_reports_nothing_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "v.csv"
    path.write_text("", encoding="utf-8")
    report = build_report(path)
    assert report.n_rows == 0
    assert report.agreements == ()


def test_oracle_ceiling_is_one_without_noise_and_half_at_coin_flip() -> None:
    from unc_bench.analysis.human_agreement import oracle_auroc_ceiling

    assert oracle_auroc_ceiling(0.0, 0.5) == 1.0
    assert oracle_auroc_ceiling(0.5, 0.5) == 0.5
    # Hand-derived: e=0.1, balanced classes -> 0.81 + 0.5*(0.09 + 0.09) = 0.90.
    assert oracle_auroc_ceiling(0.1, 0.5) == 0.9
    assert math.isnan(oracle_auroc_ceiling(1.5, 0.5))


def test_kappa_ci_is_point_at_perfect_agreement() -> None:
    from unc_bench.analysis.human_agreement import kappa_bootstrap_ci

    low, high = kappa_bootstrap_ci(
        ["correct"] * 10 + ["incorrect"] * 10, ["correct"] * 10 + ["incorrect"] * 10
    )
    assert (low, high) == (1.0, 1.0)


def test_gate_fails_honestly_without_human_labels() -> None:
    from unc_bench.analysis.validity import MIN_HUMAN_LABEL_COVERAGE, human_label_gate

    failed = human_label_gate(0.0)
    assert failed.name == "human_label_coverage"
    assert failed.passed is False
    assert "0.000" in failed.observed
    assert human_label_gate(None).passed is False
    assert human_label_gate(MIN_HUMAN_LABEL_COVERAGE).passed is True
