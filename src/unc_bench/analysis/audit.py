"""Post-hoc audit of the committed results, reading results.json only.

Three questions the published tables did not answer, all answerable from stored
values without re-running anything:

1. Which of the 21 signal rows are rank-equivalent? AUROC is invariant under a
   strictly monotone transform of the score, so a signal that is a monotone
   reparameterization of another is the same signal to every rank statistic in
   this project. The stored Spearman matrix decides this empirically.

2. What does Holm-Bonferroni look like if the rank-equivalent duplicates are
   dropped from the family? The stored per-comparison p-values are the bootstrap
   output, so this is a re-derivation from stored values rather than a new test.

3. Does any signal's PopQA AUROC clear chance? The per-dataset block stores
   point estimates with no interval, so an interval has to come from somewhere.
   `analytic_auroc_ci` is the Hanley-McNeil normal approximation, which is a
   pure function of (AUROC, n_pos, n_neg) and therefore computable from what is
   stored. It is NOT the percentile bootstrap the rest of the project reports,
   and `CI_METHOD_NOTE` exists so no caller can quote it as if it were.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Exact rank correlation required before two signals are called duplicates.
#: A monotone transform preserves ranks exactly, so anything short of 1.0 is a
#: different ordering and not a duplicate, however close it looks.
EXACT = 1.0

#: Attached to every analytic interval this module produces. The project's
#: published intervals are percentile bootstrap over 10,000 resamples; these are
#: a normal approximation and the two do not have to agree.
CI_METHOD_NOTE = (
    "Hanley-McNeil normal approximation, not the percentile bootstrap used for "
    "the pooled table; a pure function of AUROC, n_pos and n_neg"
)

#: 1.96 for a two-sided 95% normal interval. Spelled out rather than pulled from
#: scipy: this module must import nothing the CI does not already have.
Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True)
class DuplicatePair:
    """Two signals and the measured rank correlation between them."""

    a: str
    b: str
    spearman: float

    @property
    def rank_equivalent(self) -> bool:
        """True only on an exact +-1.000, which is what a monotone map gives."""
        return abs(self.spearman) == EXACT

    @property
    def verdict(self) -> str:
        if self.rank_equivalent:
            direction = "increasing" if self.spearman > 0 else "decreasing"
            return f"rank-equivalent (Spearman {self.spearman:+.3f}, monotone {direction})"
        return f"not rank-equivalent (Spearman {self.spearman:+.6f})"


@dataclass(frozen=True, slots=True)
class AnalyticCI:
    """An AUROC point estimate with an analytic interval and its provenance."""

    name: str
    auroc: float
    ci_low: float
    ci_high: float
    n_pos: int
    n_neg: int
    method: str = CI_METHOD_NOTE

    @property
    def excludes_chance(self) -> bool:
        """Whether the interval excludes 0.50 in either direction."""
        return self.ci_low > 0.5 or self.ci_high < 0.5

    @property
    def out_of_range(self) -> bool:
        """Whether an endpoint left [0, 1], which the normal approximation can do.

        AUROC is bounded on [0, 1]; a symmetric normal interval is not. An
        endpoint outside the range means the approximation has broken down for
        this subset's class counts and the interval should not be read at all.
        """
        return self.ci_low < 0.0 or self.ci_high > 1.0

    @property
    def standard_error(self) -> float:
        """The standard error implied by the interval half-width."""
        return (self.ci_high - self.ci_low) / (2.0 * Z_95)

    def chance_p_value(self) -> float:
        """Two-sided p for H0: AUROC = 0.50, from the analytic standard error.

        Unadjusted. Any signal quoted from this has been selected as the best of
        a table, so `selection_adjusted_p` is the one that answers the question
        the README asks.
        """
        se = self.standard_error
        if not math.isfinite(se) or se <= 0.0:
            return float("nan")
        return math.erfc(abs(self.auroc - 0.5) / se / math.sqrt(2.0))

    def selection_adjusted_p(self, family_size: int) -> float:
        """Bonferroni-adjusted p for having picked this signal out of a family.

        The per-dataset table is scanned for its maximum, so the largest AUROC in
        it is a selected maximum and its unadjusted p is optimistic by roughly
        the family size.
        """
        raw = self.chance_p_value()
        if not math.isfinite(raw):
            return float("nan")
        return min(1.0, raw * family_size)

    @property
    def margin_from_chance(self) -> float:
        """Signed distance from the nearer interval endpoint to 0.50.

        Positive means the interval clears 0.50. A small positive value is the
        interesting case: it means the conclusion depends on the interval method
        rather than on the data.
        """
        if self.ci_low > 0.5:
            return self.ci_low - 0.5
        if self.ci_high < 0.5:
            return 0.5 - self.ci_high
        return -min(0.5 - self.ci_low, self.ci_high - 0.5)


def load_results(path: str | Path) -> dict[str, Any]:
    """Read results.json. No other input is permitted in this module."""
    with Path(path).open(encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return payload


def spearman_between(payload: dict[str, Any], a: str, b: str, *, view: str = "primary") -> float:
    """Stored Spearman correlation between two signals over the frozen rows.

    Reads `views.<view>.correlation`, which the analysis stage computed over the
    same 120 rows every other number comes from. `None` in the matrix means the
    pair had fewer than 3 usable overlapping rows and is surfaced as NaN.
    """
    block = payload["views"][view]["correlation"]
    names: list[str] = block["names"]
    for name in (a, b):
        if name not in names:
            raise KeyError(f"{name} is not in the stored correlation matrix")
    value = block["spearman"][names.index(a)][names.index(b)]
    return float("nan") if value is None else float(value)


def check_pairs(
    payload: dict[str, Any],
    pairs: tuple[tuple[str, str], ...],
    *,
    view: str = "primary",
) -> list[DuplicatePair]:
    """Measure rank correlation for each candidate duplicate pair."""
    return [DuplicatePair(a, b, spearman_between(payload, a, b, view=view)) for a, b in pairs]


#: The pairs the published table made suspicious: each shows an identical AUROC
#: to three decimal places and has a plausible monotone relationship.
#:   a_mean_logprob / a_perplexity           exp(-x)
#:   b_distinct_count / b_distinct_fraction  divide by a constant set size
#:   b_semantic_entropy / *_normalized       divide by ln(set size); constant
#:                                           here only because the set size is
#:                                           fixed at n_samples + 1 = 6
SUSPECTED_DUPLICATE_PAIRS: tuple[tuple[str, str], ...] = (
    ("a_mean_logprob", "a_perplexity"),
    ("b_distinct_count", "b_distinct_fraction"),
    ("b_semantic_entropy", "b_semantic_entropy_normalized"),
)


def analytic_auroc_ci(
    auroc: float, n_pos: int, n_neg: int, *, z: float = Z_95
) -> tuple[float, float]:
    """Hanley-McNeil normal interval for a single AUROC.

    The variance formula assumes an exponential score distribution and no ties.
    Several signals here are integer-valued with heavy ties, which is the same
    objection that made this project substitute a paired bootstrap for DeLong in
    the significance table, so this interval is an approximation and is labelled
    as one everywhere it is used.

    Computed on `max(auroc, 1 - auroc)` and mirrored back, because the formula is
    only defined for A >= 0.5 and a sub-chance AUROC is a chance-level signal
    read in the wrong direction.
    """
    if not math.isfinite(auroc) or n_pos < 1 or n_neg < 1:
        return (float("nan"), float("nan"))
    flipped = auroc < 0.5
    a = 1.0 - auroc if flipped else auroc
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    variance = (a * (1.0 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (
        n_pos * n_neg
    )
    half = z * math.sqrt(max(variance, 0.0))
    low, high = a - half, a + half
    if flipped:
        low, high = 1.0 - high, 1.0 - low
    return (low, high)


def per_dataset_analytic_cis(
    payload: dict[str, Any],
    dataset: str,
    *,
    view: str = "primary",
) -> list[AnalyticCI]:
    """Analytic AUROC intervals for every signal on one dataset subset.

    The class counts come from the stored per-dataset block, so the interval
    width reflects that subset's size and balance rather than the pooled run's.
    Sorted by AUROC descending.
    """
    block = payload["views"][view]["per_dataset"]["datasets"][dataset]
    n_pos = int(block["n_incorrect"])
    n_neg = int(block["n_correct"])
    out: list[AnalyticCI] = []
    for name, entry in block["signals"].items():
        auroc = float(entry["auroc"])
        low, high = analytic_auroc_ci(auroc, n_pos, n_neg)
        out.append(AnalyticCI(name, auroc, low, high, n_pos, n_neg))
    out.sort(key=lambda item: (-item.auroc, item.name))
    return out


def holm(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm-Bonferroni step-down over a family of p-values.

    Returns name -> (adjusted p, significant). The running maximum enforces
    monotonicity of the adjusted values, which is what makes the step-down
    procedure coherent: a hypothesis cannot be rejected while a more extreme one
    in the same family is not.
    """
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    family = len(ordered)
    out: dict[str, tuple[float, bool]] = {}
    running = 0.0
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (family - index) * raw))
        out[name] = (running, running <= alpha)
    return out


@dataclass(frozen=True, slots=True)
class HolmComparison:
    """One signal's Holm outcome under the original and deduplicated families."""

    name: str
    p_value: float
    holm_full: float
    significant_full: bool
    holm_dedup: float | None
    significant_dedup: bool | None

    @property
    def dropped(self) -> bool:
        """True when this row was removed as a rank-equivalent duplicate."""
        return self.holm_dedup is None

    @property
    def verdict_changed(self) -> bool:
        if self.significant_dedup is None:
            return False
        return self.significant_full != self.significant_dedup


def deduplicated_holm(
    payload: dict[str, Any],
    *,
    view: str = "primary",
    drop: frozenset[str] | None = None,
) -> list[HolmComparison]:
    """Recompute Holm with the rank-equivalent duplicates removed.

    The stored p-values are reused exactly; only the family size changes. This
    is why the recomputation is legitimate without re-running the bootstrap: a
    rank-equivalent duplicate carries no information the family did not already
    have, so removing it corrects the family size rather than the test.

    When `drop` is None the duplicates are discovered from the stored Spearman
    matrix rather than hardcoded. For each rank-equivalent pair the second member
    is dropped, keeping the one whose name the published table lists first.
    """
    significance = payload["views"][view]["significance"]
    comparisons = significance["comparisons"]
    alpha = float(significance["alpha"])
    stored = {str(c["name"]): float(c["p_value"]) for c in comparisons}

    if drop is None:
        drop = frozenset(
            pair.b
            for pair in check_pairs(payload, SUSPECTED_DUPLICATE_PAIRS, view=view)
            if pair.rank_equivalent and pair.b in stored
        )

    full = holm(stored, alpha=alpha)
    reduced_input = {k: v for k, v in stored.items() if k not in drop}
    reduced = holm(reduced_input, alpha=alpha)

    out: list[HolmComparison] = []
    for name, raw in sorted(stored.items(), key=lambda item: (item[1], item[0])):
        holm_full, sig_full = full[name]
        if name in drop:
            out.append(HolmComparison(name, raw, holm_full, sig_full, None, None))
        else:
            holm_dedup, sig_dedup = reduced[name]
            out.append(HolmComparison(name, raw, holm_full, sig_full, holm_dedup, sig_dedup))
    return out


def independent_signal_count(
    payload: dict[str, Any],
    *,
    view: str = "primary",
    pairs: tuple[tuple[str, str], ...] = SUSPECTED_DUPLICATE_PAIRS,
) -> tuple[int, int]:
    """(rows in the table, distinct orderings among them).

    Only the pairs measured as rank-equivalent are collapsed. Two signals that
    merely correlate highly are still two signals.
    """
    total = len(payload["views"][view]["signals"])
    duplicates = sum(1 for pair in check_pairs(payload, pairs, view=view) if pair.rank_equivalent)
    return (total, total - duplicates)


@dataclass(frozen=True, slots=True)
class MethodAgreement:
    """How far the analytic interval sits from the bootstrap interval.

    Measured on the pooled view, the one place where both a stored bootstrap
    interval and an analytic interval exist for the same signals. It calibrates
    the analytic method against the method this project actually reports, which
    is the only way to say whether a narrow analytic margin means anything.
    """

    n_signals: int
    max_low_gap: float
    max_high_gap: float
    mean_low_gap: float
    mean_high_gap: float
    chance_verdict_disagreements: int

    @property
    def max_endpoint_gap(self) -> float:
        return max(self.max_low_gap, self.max_high_gap)


def method_agreement(payload: dict[str, Any], *, view: str = "primary") -> MethodAgreement:
    """Compare analytic intervals against the stored bootstrap on the pooled view.

    Both intervals describe the same 120 rows and the same class counts, so any
    difference is the method and nothing else.
    """
    block = payload["views"][view]
    n_pos = int(block["n_incorrect"])
    n_neg = int(block["n_correct"])
    low_gaps: list[float] = []
    high_gaps: list[float] = []
    disagreements = 0
    for entry in block["signals"].values():
        stored = entry["auroc"]
        point = float(stored["point"])
        if not math.isfinite(point):
            continue
        boot_low, boot_high = float(stored["ci_low"]), float(stored["ci_high"])
        an_low, an_high = analytic_auroc_ci(point, n_pos, n_neg)
        low_gaps.append(abs(an_low - boot_low))
        high_gaps.append(abs(an_high - boot_high))
        if (boot_low > 0.5 or boot_high < 0.5) != (an_low > 0.5 or an_high < 0.5):
            disagreements += 1
    count = len(low_gaps)
    if count == 0:
        nan = float("nan")
        return MethodAgreement(0, nan, nan, nan, nan, 0)
    return MethodAgreement(
        n_signals=count,
        max_low_gap=max(low_gaps),
        max_high_gap=max(high_gaps),
        mean_low_gap=math.fsum(low_gaps) / count,
        mean_high_gap=math.fsum(high_gaps) / count,
        chance_verdict_disagreements=disagreements,
    )


def render_report(payload: dict[str, Any], *, view: str = "primary") -> str:
    """The audit as text. Printed by `unc-bench audit`; no files are written."""
    lines: list[str] = []
    add = lines.append

    add("Rank equivalence over the frozen analysis set")
    add(f"  source: views.{view}.correlation, computed on the same rows as every other number")
    for pair in check_pairs(payload, SUSPECTED_DUPLICATE_PAIRS, view=view):
        add(f"  {pair.a} vs {pair.b}")
        add(f"    Spearman {pair.spearman:+.6f} -> {pair.verdict}")

    total, independent = independent_signal_count(payload, view=view)
    add("")
    add(f"Signal count: {total} table rows, {independent} distinct orderings")

    add("")
    add("Holm-Bonferroni, original family vs duplicates removed")
    add("  stored p-values reused; only the family size changes")
    rows = deduplicated_holm(payload, view=view)
    kept = [r for r in rows if not r.dropped]
    add(f"  family size {len(rows)} -> {len(kept)}")
    add(f"  {'signal':32s} {'p':>9s} {'holm':>9s} {'sig':>5s} {'holm*':>9s} {'sig*':>5s}")
    for row in rows:
        if row.dropped:
            add(
                f"  {row.name:32s} {row.p_value:9.6f} {row.holm_full:9.6f} "
                f"{row.significant_full!s:>5s} {'dropped':>9s} {'-':>5s}"
            )
        else:
            flag = "  <- verdict changed" if row.verdict_changed else ""
            add(
                f"  {row.name:32s} {row.p_value:9.6f} {row.holm_full:9.6f} "
                f"{row.significant_full!s:>5s} {row.holm_dedup:9.6f} "
                f"{row.significant_dedup!s:>5s}{flag}"
            )
    changed = [r.name for r in rows if r.verdict_changed]
    add(f"  verdicts changed by deduplication: {changed if changed else 'none'}")

    agreement = method_agreement(payload, view=view)
    add("")
    add("Analytic interval vs stored bootstrap, pooled view")
    add(f"  {agreement.n_signals} signals compared on the same rows and class counts")
    add(
        f"  endpoint gap: max {agreement.max_endpoint_gap:.4f}, "
        f"mean low {agreement.mean_low_gap:.4f}, mean high {agreement.mean_high_gap:.4f}"
    )
    add(f"  0.50-exclusion verdicts that disagree: {agreement.chance_verdict_disagreements}")

    for dataset in sorted(payload["views"][view]["per_dataset"]["datasets"]):
        cis = per_dataset_analytic_cis(payload, dataset, view=view)
        if not cis:
            continue
        add("")
        add(f"Per-dataset AUROC with analytic interval: {dataset}")
        add(f"  n_pos {cis[0].n_pos}, n_neg {cis[0].n_neg}")
        add(f"  {CI_METHOD_NOTE}")
        for item in cis:
            marks = []
            if item.excludes_chance:
                marks.append("excludes 0.50")
            if item.out_of_range:
                marks.append("endpoint outside [0,1]: approximation unusable")
            add(
                f"  {item.name:32s} {item.auroc:6.3f} "
                f"[{item.ci_low:6.3f}, {item.ci_high:6.3f}] {' '.join(marks)}"
            )
        clearing = [item for item in cis if item.excludes_chance]
        if clearing:
            for item in clearing:
                margin = item.margin_from_chance
                add(f"  {item.name} clears 0.50 by {margin:.4f} on the interval endpoint")
                if margin < agreement.max_endpoint_gap:
                    add(
                        f"    margin {margin:.4f} is smaller than the "
                        f"{agreement.max_endpoint_gap:.4f} disagreement between this "
                        "approximation and the bootstrap, so the exclusion is not robust "
                        "to the choice of interval method: INCONCLUSIVE"
                    )
                raw_p = item.chance_p_value()
                adjusted = item.selection_adjusted_p(independent)
                add(
                    f"    unadjusted p vs 0.50: {raw_p:.4f}; "
                    f"Bonferroni over {independent} distinct orderings: {adjusted:.4f} "
                    f"({'still below' if adjusted <= 0.05 else 'above'} 0.05)"
                )
        else:
            add("  no signal's interval excludes 0.50 on this subset")

    return "\n".join(lines)
