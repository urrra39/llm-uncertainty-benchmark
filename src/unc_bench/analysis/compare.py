"""Side-by-side AUROC comparison of two results files.

Exists so run #2 and a later, larger run can be read against each other without
anyone transcribing numbers between two JSON files by hand. It reads results
files only: no model, no dataset, no recomputation. Every number it prints is a
value stored by the analysis stage that produced the file.

Two things it deliberately does not do.

It does not test whether a difference between runs is significant. The two runs
have different row sets, so a paired test is unavailable and an unpaired one on
two bootstrap intervals would be a comparison of overlapping-interval heuristics
rather than a test. What is printed is each run's own interval, side by side, and
whether they overlap — which is a description, not an inference, and is labelled
as such.

It does not merge, average or pool the two runs. A signal that moves between
runs has moved for a reason — a different subject model, a different n, a
different dataset balance — and hiding that in a weighted mean would destroy the
only information the comparison carries.

Signals present in one file and absent from the other are reported by name
rather than dropped, because a signal that stopped being computed is a fact
about the second run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Printed instead of a number wherever a file stores null. The analysis stage
#: writes null for "not measured", and a dash keeps that distinguishable from a
#: measured zero in the rendered table.
ABSENT = "-"


def load(path: str | Path) -> dict[str, Any]:
    """Read a results file. The only input this module accepts."""
    with Path(path).open(encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return payload


def _num(value: Any) -> float:
    """Read a stored number, mapping null and unparseable values to NaN."""
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _fmt(value: float, width: int = 6, places: int = 3) -> str:
    if not math.isfinite(value):
        return f"{ABSENT:>{width}s}"
    return f"{value:{width}.{places}f}"


def _fmt_ci(low: float, high: float) -> str:
    if not (math.isfinite(low) and math.isfinite(high)):
        return f"[{ABSENT:>6s}, {ABSENT:>6s}]"
    return f"[{low:6.3f}, {high:6.3f}]"


@dataclass(frozen=True, slots=True)
class RunSummary:
    """The identifying facts about one run, for the comparison header."""

    label: str
    run_name: str
    model: str
    n: int
    n_incorrect: int
    n_correct: int
    base_rate: float
    mix: dict[str, int]
    qid_digest: str
    gates_passed: bool
    resamples: int


def summarize(payload: dict[str, Any], label: str, *, view: str = "primary") -> RunSummary:
    """Pull one run's header facts out of its results file."""
    block = payload["views"][view]
    frozen = payload.get("frozen_analysis_set", {}).get(view, {})
    gates = payload.get("validity_gates", {})
    return RunSummary(
        label=label,
        run_name=str(payload.get("run_name", "?")),
        model=str(payload.get("model_under_test", {}).get("name", "?")),
        n=int(block.get("n", 0)),
        n_incorrect=int(block.get("n_incorrect", 0)),
        n_correct=int(block.get("n_correct", 0)),
        base_rate=_num(block.get("base_rate_incorrect")),
        mix={str(k): int(v) for k, v in payload.get("dataset", {}).get("mix", {}).items()},
        qid_digest=str(frozen.get("qid_digest", "?")),
        gates_passed=bool(gates.get("all_passed", False)),
        resamples=int(payload.get("analysis_config", {}).get("bootstrap_resamples", 0)),
    )


@dataclass(frozen=True, slots=True)
class SignalComparison:
    """One signal's AUROC and interval in both runs."""

    name: str
    family: str
    in_left: bool
    in_right: bool
    left_point: float
    left_low: float
    left_high: float
    right_point: float
    right_low: float
    right_high: float

    @property
    def delta(self) -> float:
        """Right minus left. NaN when either side is missing.

        Not a treatment effect. The two runs differ in more than one respect at
        once, so this is the change in a measured quantity between two studies
        and nothing stronger.
        """
        return self.right_point - self.left_point

    @property
    def both_present(self) -> bool:
        return self.in_left and self.in_right

    @property
    def intervals_overlap(self) -> bool | None:
        """Whether the two intervals share any value.

        None when either interval is missing. Overlap is descriptive: two
        overlapping intervals do not establish equality, and two disjoint ones
        computed on different row sets are not a paired significance test.
        """
        if not all(
            math.isfinite(v)
            for v in (self.left_low, self.left_high, self.right_low, self.right_high)
        ):
            return None
        return not (self.left_high < self.right_low or self.right_high < self.left_low)

    @property
    def identical(self) -> bool:
        """Exact equality of both point estimates and both endpoints.

        Used by the self-comparison test: comparing a file against itself must
        report every signal identical and zero differences.
        """
        if not self.both_present:
            return False
        return (
            self.left_point == self.right_point
            and self.left_low == self.right_low
            and self.left_high == self.right_high
        )


def compare_signals(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    view: str = "primary",
) -> list[SignalComparison]:
    """Every signal in either run, ordered by the left run's AUROC descending.

    Ordering follows the left run so the table reads as "the earlier run's
    ranking, and what happened to it". Signals absent from the left run are
    appended alphabetically, since they have no position in that ranking.
    """
    left_signals = left["views"][view]["signals"]
    right_signals = right["views"][view]["signals"]
    catalog = {**right.get("signal_catalog", {}), **left.get("signal_catalog", {})}

    out: list[SignalComparison] = []
    for name in sorted(set(left_signals) | set(right_signals)):
        left_entry = left_signals.get(name, {}).get("auroc", {})
        right_entry = right_signals.get(name, {}).get("auroc", {})
        out.append(
            SignalComparison(
                name=name,
                family=str(catalog.get(name, {}).get("family", "?")),
                in_left=name in left_signals,
                in_right=name in right_signals,
                left_point=_num(left_entry.get("point")),
                left_low=_num(left_entry.get("ci_low")),
                left_high=_num(left_entry.get("ci_high")),
                right_point=_num(right_entry.get("point")),
                right_low=_num(right_entry.get("ci_low")),
                right_high=_num(right_entry.get("ci_high")),
            )
        )

    def sort_key(item: SignalComparison) -> tuple[int, float, str]:
        if item.in_left and math.isfinite(item.left_point):
            return (0, -item.left_point, item.name)
        return (1, 0.0, item.name)

    out.sort(key=sort_key)
    return out


def _header(summary: RunSummary) -> list[str]:
    mix = ", ".join(f"{k} {v}" for k, v in sorted(summary.mix.items()) if v)
    return [
        f"  {summary.label}: {summary.run_name}",
        f"    model         {summary.model}",
        f"    rows          n={summary.n} "
        f"({summary.n_incorrect} incorrect, {summary.n_correct} correct, "
        f"base rate {_fmt(summary.base_rate, 5, 3).strip()})",
        f"    mix           {mix or ABSENT}",
        f"    qid digest    {summary.qid_digest}",
        f"    validity      {'all gates passed' if summary.gates_passed else 'GATES FAILED'}",
        f"    bootstrap     {summary.resamples} resamples",
    ]


def render(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "run A",
    right_label: str = "run B",
    view: str = "primary",
) -> str:
    """The comparison as text. Printed by `unc-bench compare-runs`."""
    lines: list[str] = []
    add = lines.append

    left_summary = summarize(left, left_label, view=view)
    right_summary = summarize(right, right_label, view=view)

    add(f"AUROC comparison, view: {view}")
    add("")
    add("Runs")
    lines.extend(_header(left_summary))
    add("")
    lines.extend(_header(right_summary))

    same_rows = left_summary.qid_digest == right_summary.qid_digest
    add("")
    if same_rows:
        add("  Both files describe the same row set, so differences are analysis-side only.")
    else:
        add("  The two runs have different row sets. Differences below combine every way")
        add("  the runs differ; they are not attributable to any single change.")

    comparisons = compare_signals(left, right, view=view)
    add("")
    add("Per-signal AUROC with 95% bootstrap interval")
    add(
        f"  {'signal':32s} {'fam':>3s} "
        f"{'A':>6s} {'A 95% CI':>16s} "
        f"{'B':>6s} {'B 95% CI':>16s} "
        f"{'B-A':>7s}  note"
    )
    for item in comparisons:
        notes: list[str] = []
        if not item.in_left:
            notes.append(f"absent from {left_label}")
        if not item.in_right:
            notes.append(f"absent from {right_label}")
        if item.both_present:
            overlap = item.intervals_overlap
            if overlap is False:
                notes.append("intervals disjoint")
            elif overlap is True:
                notes.append("intervals overlap")
        delta = f"{item.delta:+7.3f}" if math.isfinite(item.delta) else f"{ABSENT:>7s}"
        add(
            f"  {item.name:32s} {item.family:>3s} "
            f"{_fmt(item.left_point)} {_fmt_ci(item.left_low, item.left_high):>16s} "
            f"{_fmt(item.right_point)} {_fmt_ci(item.right_low, item.right_high):>16s} "
            f"{delta}  {'; '.join(notes)}"
        )

    differing = [c for c in comparisons if not c.identical]
    add("")
    add(f"  signals compared        {len(comparisons)}")
    add(f"  present in both         {sum(1 for c in comparisons if c.both_present)}")
    add(f"  identical point and CI  {len(comparisons) - len(differing)}")
    add(f"  differing               {len(differing)}")
    if not differing:
        add("  The two files agree on every signal's point estimate and both endpoints.")
    add("")
    add("  Interval overlap is descriptive. The runs are scored on different rows, so")
    add("  no paired test between them is available and none is reported.")
    return "\n".join(lines)


def count_differences(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    view: str = "primary",
) -> int:
    """How many signals differ in point estimate or either endpoint.

    Zero for a file compared against itself. That is the property the test
    suite pins, and it is the reason this is a function rather than a number
    parsed back out of the rendered text.
    """
    return sum(1 for c in compare_signals(left, right, view=view) if not c.identical)


def run(
    left_path: str | Path,
    right_path: str | Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
    view: str = "primary",
) -> int:
    """Load both files, print the comparison, return the number of differences."""
    left = load(left_path)
    right = load(right_path)
    print(
        render(
            left,
            right,
            left_label=left_label or str(left_path),
            right_label=right_label or str(right_path),
            view=view,
        )
    )
    return count_differences(left, right, view=view)
