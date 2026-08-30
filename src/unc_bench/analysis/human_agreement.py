"""Judge-versus-human agreement over a filled-in validation sample.

Nothing in the published run used this. Every label in results.json is model
assigned, and the kappa quoted there is judge against judge, which measures
whether two judges agree with each other and says nothing about whether either
agrees with a human. This module is the path from a filled-in
`data/human_validation_sample.csv` to that missing number.

It reads a CSV and computes agreement. It does not call a model, does not write
files, and does not touch results.json. Rows with an empty `human_label` are
skipped and counted, so running it on the shipped file — where the column is
empty by construction, because inventing labels would be fabrication — reports
zero labelled rows rather than a fabricated agreement.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from unc_bench.labeling import cohens_kappa

#: The two verdicts a labeller may write. `abstain` is deliberately absent: run
#: #2 recorded 0 abstentions, and a human who thinks a row is unscoreable should
#: leave it blank so it lands in `n_unlabelled` rather than inventing a third
#: category the judges were never offered.
HUMAN_LABELS = ("correct", "incorrect")

#: Column holding the human verdict. Empty means "not yet labelled".
HUMAN_COLUMN = "human_label"

#: Columns compared against the human, in report order. Each is optional in the
#: sense that a column present but empty on a row makes that row uncomparable
#: for that comparison only, not for the others.
MACHINE_COLUMNS = (
    "heuristic_verdict",
    "judge_primary_verdict",
    "judge_secondary_verdict",
    "machine_label",
)


@dataclass(frozen=True, slots=True)
class Agreement:
    """One machine column scored against the human column."""

    machine_column: str
    n_compared: int
    n_agree: int
    kappa: float
    observed_agreement: float
    expected_agreement: float
    kappa_available: bool
    kappa_note: str

    @property
    def disagreements(self) -> int:
        return self.n_compared - self.n_agree


@dataclass(frozen=True, slots=True)
class HumanValidationReport:
    """Everything derivable from a validation CSV, with the gaps counted."""

    path: Path
    n_rows: int
    n_labelled: int
    n_unlabelled: int
    invalid_labels: tuple[tuple[str, str], ...]
    agreements: tuple[Agreement, ...]

    @property
    def usable(self) -> bool:
        return self.n_labelled > 0 and not self.invalid_labels


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def read_validation_csv(path: str | Path) -> list[dict[str, str]]:
    """Read the validation sample, preserving every column as text."""
    target = Path(path)
    with target.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def score_column(rows: list[dict[str, str]], machine_column: str) -> Agreement | None:
    """Agreement between one machine column and the human column.

    Returns None when the column is absent from the file entirely, which is
    different from present-but-unlabelled and should not be reported as 0.
    """
    if not rows or machine_column not in rows[0]:
        return None

    paired: list[tuple[str, str]] = []
    for row in rows:
        human = _normalize(row.get(HUMAN_COLUMN))
        machine = _normalize(row.get(machine_column))
        if human in HUMAN_LABELS and machine in HUMAN_LABELS:
            paired.append((machine, human))

    if not paired:
        return Agreement(
            machine_column=machine_column,
            n_compared=0,
            n_agree=0,
            kappa=float("nan"),
            observed_agreement=float("nan"),
            expected_agreement=float("nan"),
            kappa_available=False,
            kappa_note="no row has both a human label and this machine verdict",
        )

    machine_side = [m for m, _ in paired]
    human_side = [h for _, h in paired]
    n_agree = sum(1 for m, h in paired if m == h)

    result = cohens_kappa(machine_side, human_side)
    note = ""
    if not result.trustworthy:
        note = (
            "kappa is undefined when both sides use a single category with no "
            "variance to measure"
            if result.expected_agreement >= 1.0
            else "kappa below the project's trust threshold"
        )
    return Agreement(
        machine_column=machine_column,
        n_compared=len(paired),
        n_agree=n_agree,
        kappa=result.kappa,
        observed_agreement=result.observed_agreement,
        expected_agreement=result.expected_agreement,
        kappa_available=result.trustworthy,
        kappa_note=note,
    )


def build_report(path: str | Path) -> HumanValidationReport:
    """Score every machine column in the file against the human column."""
    target = Path(path)
    rows = read_validation_csv(target)

    if rows and HUMAN_COLUMN not in rows[0]:
        raise KeyError(f"{target} has no '{HUMAN_COLUMN}' column")

    labelled = 0
    invalid: list[tuple[str, str]] = []
    for row in rows:
        raw = _normalize(row.get(HUMAN_COLUMN))
        if not raw:
            continue
        if raw in HUMAN_LABELS:
            labelled += 1
        else:
            invalid.append((row.get("qid", "?"), row.get(HUMAN_COLUMN, "")))

    agreements = tuple(
        scored for column in MACHINE_COLUMNS if (scored := score_column(rows, column)) is not None
    )
    return HumanValidationReport(
        path=target,
        n_rows=len(rows),
        n_labelled=labelled,
        n_unlabelled=len(rows) - labelled - len(invalid),
        invalid_labels=tuple(invalid),
        agreements=agreements,
    )


def render_report(report: HumanValidationReport) -> str:
    """The report as text. Printed by `unc-bench human-agreement`."""
    lines: list[str] = []
    add = lines.append

    add(f"Human validation: {report.path}")
    add(f"  rows {report.n_rows}, labelled {report.n_labelled}, unlabelled {report.n_unlabelled}")

    if report.invalid_labels:
        add(f"  rejected {len(report.invalid_labels)} row(s) with an unrecognised label:")
        for qid, value in report.invalid_labels[:10]:
            add(f"    {qid}: {value!r} is not one of {HUMAN_LABELS}")
        add("  fix these before reading any agreement below")

    if report.n_labelled == 0:
        add("")
        add("  No human labels present, so no agreement can be computed.")
        add("  This is the shipped state of the file: the labels are for a human to")
        add("  fill in, and an agreement computed without them would be fabricated.")
        add("  See data/README.md for the labelling convention.")
        return "\n".join(lines)

    for scored in report.agreements:
        add("")
        add(f"  {scored.machine_column} vs {HUMAN_COLUMN}")
        if scored.n_compared == 0:
            add(f"    {scored.kappa_note}")
            continue
        rate = scored.n_agree / scored.n_compared
        add(
            f"    agreement {scored.n_agree}/{scored.n_compared} = {rate:.4f} "
            f"({scored.disagreements} disagreement(s))"
        )
        if scored.kappa != scored.kappa:  # NaN
            add(f"    Cohen's kappa: undefined - {scored.kappa_note}")
        else:
            add(f"    Cohen's kappa {scored.kappa:.4f}")
            if scored.kappa_note:
                add(f"    note: {scored.kappa_note}")
        add(f"    expected agreement by chance {scored.expected_agreement:.4f}")

    add("")
    add("  This measures label correctness against a human, which is a different")
    add("  quantity from the judge-versus-judge kappa in results.json.")
    return "\n".join(lines)


def run(path: str | Path) -> HumanValidationReport:
    """Build the report and print it. Returns it for tests."""
    report = build_report(path)
    print(render_report(report))
    return report
