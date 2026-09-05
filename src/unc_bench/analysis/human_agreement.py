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
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unc_bench.labeling import MIN_MINORITY_FOR_TRUST, cohens_kappa

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
    #: 95% bootstrap interval over paired resamples, NaN when uncomputable.
    kappa_ci_low: float = float("nan")
    kappa_ci_high: float = float("nan")
    #: Pooled minority-category assignments across both sides.
    minority_count: int = 0

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
    #: Fraction of labelled rows the human marked incorrect. NaN when unlabelled.
    human_base_rate_incorrect: float = float("nan")
    #: 1 - machine_label agreement rate. NaN when uncompared.
    machine_error_rate: float = float("nan")
    #: Oracle ceiling at the measured error/base rates. NaN when uncomputable.
    oracle_ceiling: float = float("nan")

    @property
    def usable(self) -> bool:
        return self.n_labelled > 0 and not self.invalid_labels


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def kappa_bootstrap_ci(
    machine: list[str],
    human: list[str],
    *,
    resamples: int = 2000,
    seed: int = 20260202,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for Cohen's κ over paired resamples.

    Rows are resampled with replacement and κ recomputed per resample; NaN
    draws (unanimous resamples) are dropped. Returns (NaN, NaN) when no finite
    draw exists. The interval says how stable the point estimate is, which is
    exactly what the recovered-κ file lacked at 53 rows and zero disagreements.
    """
    if len(machine) != len(human):
        raise ValueError(f"kappa CI needs paired labels: got {len(machine)} and {len(human)}")
    n = len(machine)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        take = rng.integers(0, n, size=n)
        value = cohens_kappa([machine[int(i)] for i in take], [human[int(i)] for i in take]).kappa
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    alpha = (1.0 - level) / 2.0
    return (float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha)))


def oracle_auroc_ceiling(error_rate: float, base_rate: float) -> float:
    """Highest AUROC observable against noisy labels, for an oracle signal.

    Closed form under symmetric independent label flips: each true label is
    flipped with probability `error_rate`, and the oracle scores the TRUE
    label (1 for incorrect, 0 for correct), so every shortfall from 1.0 is
    label noise rather than signal weakness. With π the true-positive rate
    and π̃ = (1-e)π + e(1-π) the observed one, writing p1 = P(S=1|Ỹ=1) and
    q0 = P(S=0|Ỹ=0), the midrank AUROC is p1·q0 + ½(p1·q1 + p0·q0).

    Symmetric flips are the OPTIMISTIC noise model — real judge errors
    concentrate on hard rows, which attenuates more — so this ceiling is an
    upper bound on what any signal can display. If even the ceiling sits near
    0.5, "no signal beats chance" is not a testable claim at that noise level.
    Returns NaN outside (error_rate, base_rate) in [0,1].
    """
    e, pi = error_rate, base_rate
    if not (0.0 <= e <= 1.0 and 0.0 < pi < 1.0):
        return float("nan")
    pi_tilde = (1.0 - e) * pi + e * (1.0 - pi)
    if not 0.0 < pi_tilde < 1.0:
        return float("nan")
    p1 = (1.0 - e) * pi / pi_tilde
    p0 = 1.0 - p1
    q0 = (1.0 - e) * (1.0 - pi) / (1.0 - pi_tilde)
    q1 = 1.0 - q0
    return p1 * q0 + 0.5 * (p1 * q1 + p0 * q0)


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

    result = cohens_kappa(machine_side, human_side, min_minority=MIN_MINORITY_FOR_TRUST)
    ci_low, ci_high = kappa_bootstrap_ci(machine_side, human_side)
    note = ""
    if not result.trustworthy:
        note = (
            "kappa is undefined when both sides use a single category with no "
            "variance to measure"
            if result.expected_agreement >= 1.0
            else (
                f"fewer than {MIN_MINORITY_FOR_TRUST} pooled minority assignments; "
                "one row flip moves kappa substantially"
                if result.minority_count < MIN_MINORITY_FOR_TRUST
                else "kappa below the project's trust threshold"
            )
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
        kappa_ci_low=ci_low,
        kappa_ci_high=ci_high,
        minority_count=result.minority_count,
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
    human_marks = [_normalize(row.get(HUMAN_COLUMN)) for row in rows]
    labelled_marks = [m for m in human_marks if m in HUMAN_LABELS]
    human_base = (
        sum(1 for m in labelled_marks if m == "incorrect") / len(labelled_marks)
        if labelled_marks
        else float("nan")
    )
    machine_entry = next((a for a in agreements if a.machine_column == "machine_label"), None)
    machine_error = (
        (machine_entry.disagreements / machine_entry.n_compared)
        if machine_entry is not None and machine_entry.n_compared > 0
        else float("nan")
    )
    ceiling = oracle_auroc_ceiling(machine_error, human_base)
    return HumanValidationReport(
        path=target,
        n_rows=len(rows),
        n_labelled=labelled,
        n_unlabelled=len(rows) - labelled - len(invalid),
        invalid_labels=tuple(invalid),
        agreements=agreements,
        human_base_rate_incorrect=human_base,
        machine_error_rate=machine_error,
        oracle_ceiling=ceiling,
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
            add(
                f"    Cohen's kappa {scored.kappa:.4f} "
                f"[{scored.kappa_ci_low:.4f}, {scored.kappa_ci_high:.4f}]"
            )
            if scored.kappa_note:
                add(f"    note: {scored.kappa_note}")
        add(f"    expected agreement by chance {scored.expected_agreement:.4f}")
        add(f"    pooled minority assignments {scored.minority_count}")

    add("")
    add("  This measures label correctness against a human, which is a different")
    add("  quantity from the judge-versus-judge kappa in results.json.")
    if report.oracle_ceiling == report.oracle_ceiling:  # not NaN
        add("")
        add(
            f"  Label-noise ceiling: at machine error rate {report.machine_error_rate:.4f} "
            f"and human base rate {report.human_base_rate_incorrect:.4f}, even an oracle "
            f"signal scores at most AUROC {report.oracle_ceiling:.4f} against these "
            "labels (symmetric-flip model: optimistic, so the true ceiling is lower)."
        )
    return "\n".join(lines)


def run(path: str | Path) -> HumanValidationReport:
    """Build the report and print it. Returns it for tests."""
    report = build_report(path)
    print(render_report(report))
    return report
