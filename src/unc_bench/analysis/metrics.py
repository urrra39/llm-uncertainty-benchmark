"""The metrics the results table is made of.

Written against numpy directly rather than pulled from sklearn, for one reason
that matters and one that does not. The one that matters: every function here has
to have a defined answer on the degenerate inputs this study actually produces —
a signal column that is constant, a column that is all NaN, a subset with one
class only — and returning NaN with a recorded usable-n is the behaviour I want
rather than an exception halfway through a bootstrap. The one that does not:
these are twenty lines each.

Orientation is assumed already applied. Every value reaching this module means
"higher = more likely wrong", and the positive class is INCORRECT. If that
convention were violated the AUROCs would read as 1-x, which is exactly the
failure `signals.base` exists to prevent, so nothing here re-derives it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def usable_mask(scores: FloatArray) -> BoolArray:
    """Rows where the signal has a finite value.

    NaN marks "not measured" throughout this project and infinities cannot
    appear (family A clamps perplexity), but both are excluded here rather than
    trusted, because one non-finite value silently poisons a mean, a rank and
    every bootstrap resample that draws it.
    """
    return np.isfinite(scores)


def auroc(scores: FloatArray, labels: BoolArray) -> float:
    """AUROC via the Mann-Whitney U statistic, with midranks for ties.

    Ties are the reason for the rank formulation. `t_answer_length` takes maybe
    eight distinct values over the whole study and `b_distinct_count` takes six,
    so ties are not an edge case here, they are most of the data. A
    threshold-sweeping implementation that breaks ties by input order reports a
    different number depending on how the rows were sorted; midranks give the
    probabilistic definition, which is what AUROC means.

    Returns NaN when either class is empty. A one-class subset has no AUROC, and
    0.5 would read as a measured coin flip.
    """
    if scores.size != labels.size:
        raise ValueError(f"auroc needs paired arrays: got {scores.size} and {labels.size}")
    keep = usable_mask(scores)
    s = scores[keep]
    y = labels[keep]
    n_pos = int(np.count_nonzero(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # Average rank over the tied block, 1-based.
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = float(ranks[y].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


@dataclass(frozen=True, slots=True)
class CI:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    n: int
    resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "ci_width": self.width,
            "n": self.n,
            "resamples": self.resamples,
        }


def _stratified_cluster_draw(
    y: BoolArray,
    cluster_ids: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
    """One stratified resample of row indices, drawn at the cluster level.

    Clusters are labelled by majority vote of their member rows and resampled
    with replacement within each class stratum, preserving the observed number
    of clusters per class. Rows of a drawn cluster all enter together, so
    perfectly correlated rows (identical prompts under greedy decoding) cannot
    be split across the resample. With singleton clusters this reduces exactly
    to the row-level stratified draw given the same seed.
    """
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    votes = np.zeros(len(clusters), dtype=np.float64)
    counts = np.zeros(len(clusters), dtype=np.int64)
    for position, flag in enumerate(y):
        votes[inverse[position]] += 1.0 if flag else -1.0
        counts[inverse[position]] += 1
    positive = np.flatnonzero(votes >= 0.0)
    negative = np.flatnonzero(votes < 0.0)
    if positive.size == 0 or negative.size == 0:
        # Degenerate clustering (every cluster votes one way). Fall back to the
        # row-level stratified draw rather than crashing in `integers(0, 0)`.
        pos_idx = np.flatnonzero(y)
        neg_idx = np.flatnonzero(~y)
        take_pos = rng.integers(0, pos_idx.size, size=pos_idx.size)
        take_neg = rng.integers(0, neg_idx.size, size=neg_idx.size)
        return np.concatenate([pos_idx[take_pos], neg_idx[take_neg]])
    take_pos = rng.integers(0, positive.size, size=positive.size)
    take_neg = rng.integers(0, negative.size, size=negative.size)
    chosen = np.concatenate([positive[take_pos], negative[take_neg]])
    members: list[npt.NDArray[np.int64]] = [np.flatnonzero(inverse == int(c)) for c in chosen]
    return np.concatenate(members) if members else np.zeros(0, dtype=np.int64)


def bootstrap_auroc_ci(
    scores: FloatArray,
    labels: BoolArray,
    *,
    resamples: int,
    seed: int,
    level: float = 0.95,
    cluster_ids: npt.NDArray[np.int64] | None = None,
) -> CI:
    """Stratified percentile bootstrap CI for AUROC.

    Stratified: positives and negatives are resampled within class so every
    resample keeps the observed base rate. An unstratified bootstrap at n=150
    with a 40% error rate will occasionally draw a resample that is all one
    class, whose AUROC is undefined; dropping those resamples biases the interval
    and keeping them as NaN destroys it. Stratifying makes the degenerate draw
    impossible by construction.

    Percentile rather than BCa. At these n the bias correction is a refinement on
    an interval that is already ±0.07 wide, and the percentile interval is the
    one whose coverage properties I can state without qualification.
    """
    keep = usable_mask(scores)
    s = scores[keep]
    y = labels[keep]
    point = auroc(s, y)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return CI(point=point, low=float("nan"), high=float("nan"), n=int(s.size), resamples=0)
    if cluster_ids is not None and cluster_ids.shape[0] != scores.shape[0]:
        raise ValueError(
            f"cluster bootstrap needs one id per input row: got {cluster_ids.shape[0]} "
            f"for {scores.shape[0]} rows"
        )
    clusters = cluster_ids[keep] if cluster_ids is not None else None
    if clusters is not None and clusters.shape != y.shape:
        raise ValueError(
            f"cluster bootstrap needs one id per usable row: got {clusters.shape} for {y.shape}"
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        if clusters is not None:
            idx = _stratified_cluster_draw(y, clusters, rng)
        else:
            take_pos = rng.integers(0, pos_idx.size, size=pos_idx.size)
            take_neg = rng.integers(0, neg_idx.size, size=neg_idx.size)
            idx = np.concatenate([pos_idx[take_pos], neg_idx[take_neg]])
        draws[r] = auroc(s[idx], y[idx])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:  # pragma: no cover - stratification prevents this
        return CI(point=point, low=float("nan"), high=float("nan"), n=int(s.size), resamples=0)
    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(finite, alpha))
    high = float(np.quantile(finite, 1.0 - alpha))
    return CI(point=point, low=low, high=high, n=int(s.size), resamples=int(finite.size))


@dataclass(frozen=True, slots=True)
class RiskCoverage:
    """A selective-prediction curve and its area."""

    coverage: list[float]
    risk: list[float]
    aurc: float
    base_rate: float
    coverage_at_target: float | None
    target_accuracy: float

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "risk": self.risk,
            "aurc": self.aurc,
            "base_rate": self.base_rate,
            "coverage_at_target_accuracy": self.coverage_at_target,
            "target_accuracy": self.target_accuracy,
        }


def risk_coverage(
    scores: FloatArray,
    labels: BoolArray,
    *,
    target_accuracy: float = 0.9,
) -> RiskCoverage:
    """Risk as a function of coverage, answering in increasing order of risk.

    The operational reading of the whole benchmark. Sort by the signal ascending,
    answer the first k, and the risk at coverage k/n is the error rate among
    those k. AURC is the mean risk over all k, which is the standard trapezoid-free
    definition and is comparable across signals at fixed n.

    `coverage_at_target` is the largest coverage whose risk stays at or below
    `1 - target_accuracy`. None when no prefix qualifies, which is the honest
    answer when the base rate already exceeds the target — and at a 40-60% error
    rate against a 90% accuracy target, that is a live possibility rather than a
    hypothetical.
    """
    keep = usable_mask(scores)
    s = scores[keep]
    y = labels[keep]
    n = int(s.size)
    if n == 0 or np.count_nonzero(y) == 0 or np.count_nonzero(~y) == 0:
        return RiskCoverage(
            coverage=[],
            risk=[],
            aurc=float("nan"),
            base_rate=float(np.mean(y)) if n else float("nan"),
            coverage_at_target=None,
            target_accuracy=target_accuracy,
        )
    order = np.argsort(s, kind="mergesort")
    errors = y[order].astype(np.float64)
    cumulative = np.cumsum(errors)
    ks = np.arange(1, n + 1, dtype=np.float64)
    risk = cumulative / ks
    coverage = ks / n
    max_risk = 1.0 - target_accuracy
    qualifying = np.flatnonzero(risk <= max_risk)
    at_target = float(coverage[qualifying[-1]]) if qualifying.size else None
    return RiskCoverage(
        coverage=[float(c) for c in coverage],
        risk=[float(r) for r in risk],
        aurc=float(np.mean(risk)),
        base_rate=float(np.mean(y)),
        coverage_at_target=at_target,
        target_accuracy=target_accuracy,
    )


@dataclass(frozen=True, slots=True)
class Calibration:
    """Binned reliability of a probability estimate."""

    ece: float
    bins: int
    bin_centers: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_counts: list[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "ece": self.ece,
            "bins": self.bins,
            "bin_centers": self.bin_centers,
            "bin_confidence": self.bin_confidence,
            "bin_accuracy": self.bin_accuracy,
            "bin_counts": self.bin_counts,
        }


def expected_calibration_error(
    probabilities: FloatArray,
    outcomes: BoolArray,
    *,
    bins: int = 10,
) -> Calibration:
    """ECE with equal-width bins over [0, 1].

    `probabilities` are P(incorrect) and `outcomes` are the incorrect indicator,
    so a well-calibrated signal has bin confidence equal to bin accuracy.

    Empty bins contribute nothing and are reported with a count of zero rather
    than being interpolated. At n=150 over 10 bins the average bin holds 15 items
    and several will be empty; drawing a reliability curve through invented
    points in those bins would be the most misleading thing in the whole figure
    set.
    """
    keep = usable_mask(probabilities)
    p = np.clip(probabilities[keep], 0.0, 1.0)
    y = outcomes[keep].astype(np.float64)
    if p.size == 0:
        return Calibration(float("nan"), bins, [], [], [], [])
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=True` on all but the first bin so the interval is (lo, hi] and
    # p == 1.0 lands in the last bin instead of falling off the end.
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    centers: list[float] = []
    confidence: list[float] = []
    accuracy: list[float] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(bins):
        in_bin = index == b
        count = int(np.count_nonzero(in_bin))
        centers.append(float(0.5 * (edges[b] + edges[b + 1])))
        counts.append(count)
        if count == 0:
            confidence.append(float("nan"))
            accuracy.append(float("nan"))
            continue
        conf = float(np.mean(p[in_bin]))
        acc = float(np.mean(y[in_bin]))
        confidence.append(conf)
        accuracy.append(acc)
        ece += (count / p.size) * abs(acc - conf)
    return Calibration(
        ece=ece,
        bins=bins,
        bin_centers=centers,
        bin_confidence=confidence,
        bin_accuracy=accuracy,
        bin_counts=counts,
    )


def fit_platt(scores: FloatArray, labels: BoolArray, *, max_iter: int = 200) -> tuple[float, float]:
    """Fit P(incorrect) = sigmoid(a * score + b) by Newton steps on the log loss.

    Platt scaling exists here because none of these signals is a probability. A
    logprob is not a probability of error and neither is a semantic entropy in
    nats, so an ECE computed on the raw value would measure the unit mismatch
    rather than the calibration. Fitting the two-parameter map on the TRAIN split
    only and reporting ECE on the held-out rest is what makes the number mean
    something.

    Standardizes the score first. Raw inputs here range from ~0 to ~5e8
    (clamped perplexity), and Newton's method on an unscaled feature of that
    magnitude produces a step that overflows on the first iteration.
    """
    keep = usable_mask(scores)
    s = scores[keep].astype(np.float64)
    y = labels[keep].astype(np.float64)
    if s.size == 0 or np.count_nonzero(y) == 0 or np.count_nonzero(1 - y) == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(s))
    sigma = float(np.std(s))
    if sigma <= 0.0:
        # Constant signal: the best available map is the base rate, expressed as
        # a zero-slope logit.
        rate = float(np.mean(y))
        rate = min(max(rate, 1e-6), 1 - 1e-6)
        return 0.0, float(np.log(rate / (1 - rate)))
    z = (s - mu) / sigma
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        # Clipped before the exponential. On a well-separated signal Newton
        # drives the slope up until some |logit| exceeds ~709, where exp()
        # overflows to inf; the resulting p is still the correct 0 or 1 but the
        # step is computed from inf-derived weights and numpy warns on every
        # iteration. Clipping at 60 leaves p within 1e-26 of its limit, which is
        # far tighter than the convergence tolerance below.
        logits = np.clip(a * z + b, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        w = np.clip(p * (1 - p), 1e-12, None)
        grad = np.array([float(np.sum((p - y) * z)), float(np.sum(p - y))])
        hess = np.array(
            [
                [float(np.sum(w * z * z)), float(np.sum(w * z))],
                [float(np.sum(w * z)), float(np.sum(w))],
            ]
        )
        try:
            step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        except np.linalg.LinAlgError:  # pragma: no cover - ridge prevents this
            break
        a -= float(step[0])
        b -= float(step[1])
        if float(np.max(np.abs(step))) < 1e-9:
            break
    # Fold the standardization back into (a, b) so the caller can apply the map
    # to raw scores without carrying mu and sigma around.
    return a / sigma, b - a * mu / sigma


def apply_platt(scores: FloatArray, a: float, b: float) -> FloatArray:
    """Map raw scores to probabilities with a fitted Platt map."""
    if not np.isfinite(a) or not np.isfinite(b):
        return np.full(scores.shape, np.nan, dtype=np.float64)
    out = np.full(scores.shape, np.nan, dtype=np.float64)
    keep = usable_mask(scores)
    # Clipped before exp: a large |logit| is fine mathematically but overflows
    # float64 in the exponential and emits a RuntimeWarning per row.
    logits = np.clip(a * scores[keep] + b, -60.0, 60.0)
    out[keep] = 1.0 / (1.0 + np.exp(-logits))
    return out


def spearman_matrix(columns: dict[str, FloatArray]) -> tuple[list[str], FloatArray]:
    """Pairwise Spearman correlation, computed on each pair's usable overlap.

    Spearman rather than Pearson because these signals live on wildly different
    scales and several are monotone transforms of each other; the rank
    correlation is what answers "are these two measuring the same thing".

    Ranked per pair on the shared usable rows, not once globally. Family B is
    NaN on any row its pass did not reach, and ranking a column once over its own
    usable set then correlating two differently-masked rank vectors compares
    ranks computed over different populations.
    """
    names = list(columns)
    size = len(names)
    out = np.full((size, size), np.nan, dtype=np.float64)
    for i in range(size):
        for j in range(i, size):
            a = columns[names[i]]
            b = columns[names[j]]
            both = usable_mask(a) & usable_mask(b)
            if int(np.count_nonzero(both)) < 3:
                continue
            rho = _spearman(a[both], b[both])
            out[i, j] = rho
            out[j, i] = rho
    return names, out


def _spearman(a: FloatArray, b: FloatArray) -> float:
    ra = _midranks(a)
    rb = _midranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0.0:
        # At least one column is constant on the overlap, so no monotone
        # relationship is defined. NaN, not 0.0.
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def _midranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def average_precision(scores: FloatArray, labels: BoolArray) -> float:
    """AUPRC as average precision, the step-wise sum, not trapezoidal.

    Average precision rather than the trapezoid under an interpolated PR curve.
    Interpolating precision between thresholds invents operating points that do
    not exist, and at n=120 with ties on integer-valued signals the invented
    points move the number more than the data does. The step-wise sum
    sum_k (R_k - R_{k-1}) * P_k is the one that equals the expected precision
    over the observed positives.

    Ties get one shared operating point rather than one per row. A signal like
    `b_distinct_count` takes six values, so an implementation that walks rows
    individually would credit it with a precision it cannot deliver: at
    prediction time you cannot break a tie you cannot see.

    AUPRC is reported next to AUROC because the two answer different questions
    and the study needs both. AUROC is base-rate independent, so it survives a
    change in the error rate; AUPRC is not, so it says what a triage queue built
    on the signal would actually feel like at this error rate.

    Returns NaN when no positives are present. The baseline value is the base
    rate itself, not 0.5.
    """
    if scores.size != labels.size:
        raise ValueError(
            f"average_precision needs paired arrays: got {scores.size} and {labels.size}"
        )
    keep = usable_mask(scores)
    s = scores[keep]
    y = labels[keep]
    n_pos = int(np.count_nonzero(y))
    if n_pos == 0 or s.size == 0:
        return float("nan")

    # Descending score order: the highest-risk row is retrieved first.
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.float64)
    tp = np.cumsum(y_sorted)
    retrieved = np.arange(1, s.size + 1, dtype=np.float64)

    # Keep only the last index of each tied block: a threshold can only be set
    # between distinct score values.
    is_block_end = np.ones(s.size, dtype=bool)
    is_block_end[:-1] = s_sorted[:-1] != s_sorted[1:]
    idx = np.flatnonzero(is_block_end)

    precision = tp[idx] / retrieved[idx]
    recall = tp[idx] / float(n_pos)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def bootstrap_average_precision_ci(
    scores: FloatArray,
    labels: BoolArray,
    *,
    resamples: int,
    seed: int,
    level: float = 0.95,
    cluster_ids: npt.NDArray[np.int64] | None = None,
) -> CI:
    """Stratified percentile bootstrap CI for average precision.

    Stratified for the same reason as the AUROC interval: a resample with no
    positives has no average precision. Stratifying also holds the base rate
    fixed across resamples, which matters more here than for AUROC, because
    average precision moves with the base rate and an interval that let it
    wander would be measuring two things at once.
    """
    keep = usable_mask(scores)
    s = scores[keep]
    y = labels[keep]
    point = average_precision(s, y)
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return CI(point=point, low=float("nan"), high=float("nan"), n=int(s.size), resamples=0)
    if cluster_ids is not None and cluster_ids.shape[0] != scores.shape[0]:
        raise ValueError(
            f"cluster bootstrap needs one id per input row: got {cluster_ids.shape[0]} "
            f"for {scores.shape[0]} rows"
        )
    clusters = cluster_ids[keep] if cluster_ids is not None else None
    if clusters is not None and clusters.shape != y.shape:
        raise ValueError(
            f"cluster bootstrap needs one id per usable row: got {clusters.shape} for {y.shape}"
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        if clusters is not None:
            idx = _stratified_cluster_draw(y, clusters, rng)
        else:
            take_pos = rng.integers(0, pos_idx.size, size=pos_idx.size)
            take_neg = rng.integers(0, neg_idx.size, size=neg_idx.size)
            idx = np.concatenate([pos_idx[take_pos], neg_idx[take_neg]])
        draws[r] = average_precision(s[idx], y[idx])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:  # pragma: no cover - stratification prevents this
        return CI(point=point, low=float("nan"), high=float("nan"), n=int(s.size), resamples=0)
    alpha = (1.0 - level) / 2.0
    return CI(
        point=point,
        low=float(np.quantile(finite, alpha)),
        high=float(np.quantile(finite, 1.0 - alpha)),
        n=int(s.size),
        resamples=int(finite.size),
    )


@dataclass(frozen=True, slots=True)
class PairedAurocTest:
    """One signal compared against the reference signal on identical rows."""

    name: str
    auroc: float
    reference_auroc: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_holm: float
    significant_holm: bool
    n_paired: int
    resamples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "auroc": self.auroc,
            "reference_auroc": self.reference_auroc,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "p_value_holm": self.p_value_holm,
            "significant_holm": self.significant_holm,
            "n_paired": self.n_paired,
            "resamples": self.resamples,
        }


def paired_bootstrap_auroc_diff(
    reference: FloatArray,
    other: FloatArray,
    labels: BoolArray,
    *,
    resamples: int,
    seed: int,
    level: float = 0.95,
    cluster_ids: npt.NDArray[np.int64] | None = None,
) -> tuple[float, float, float, float, int]:
    """Paired bootstrap on the AUROC difference between two correlated signals.

    Returns (delta, ci_low, ci_high, two_sided_p, n_paired).

    This is the substitute for DeLong's test, and the substitution is deliberate
    rather than a shortcut. DeLong gives an analytic variance for the difference
    of two correlated AUROCs via the structural components of the U statistic,
    and its published derivation assumes no ties in the scores. Half the signals
    here are integer-valued — `b_distinct_count` takes six values across the
    whole study, `t_answer_length` about eight — so a large share of pairs are
    tied and the midrank correction that keeps DeLong usable under ties is
    exactly the part that is easy to get subtly wrong. A paired bootstrap makes
    no distributional assumption, handles ties by construction because it just
    recomputes the tie-aware AUROC on each resample, and costs a few seconds at
    this n. README names this test rather than claiming DeLong.

    Paired means the same resample indices are applied to both signals. That is
    the whole point: the two signals are measured on the same rows and are
    strongly correlated, so an unpaired comparison would attribute their shared
    row-level noise to the difference between them and produce an interval far
    too wide.

    Rows where either signal is missing are dropped from both, so the comparison
    is on the intersection of usable rows and the two AUROCs inside it are
    recomputed on that intersection rather than reused from the full-column fit.

    The p-value is the bootstrap two-sided achieved significance level: the
    proportion of resamples whose delta falls on the opposite side of zero from
    the observed delta, doubled, with the standard +1 correction so it can never
    be exactly zero. A test on 10,000 resamples cannot resolve p below about
    1e-4 and reporting 0.0 would overstate it.
    """
    if not (reference.size == other.size == labels.size):
        raise ValueError("paired_bootstrap_auroc_diff needs three equal-length arrays")
    keep = usable_mask(reference) & usable_mask(other)
    a = reference[keep]
    b = other[keep]
    y = labels[keep]
    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    n_paired = int(y.size)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), n_paired)

    delta = auroc(b, y) - auroc(a, y)
    if cluster_ids is not None and cluster_ids.shape[0] != reference.shape[0]:
        raise ValueError(
            f"cluster bootstrap needs one id per input row: got {cluster_ids.shape[0]} "
            f"for {reference.shape[0]} rows"
        )
    clusters = cluster_ids[keep] if cluster_ids is not None else None
    if clusters is not None and clusters.shape != y.shape:
        raise ValueError(
            f"cluster bootstrap needs one id per usable row: got {clusters.shape} for {y.shape}"
        )
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for r in range(resamples):
        if clusters is not None:
            idx = _stratified_cluster_draw(y, clusters, rng)
        else:
            take_pos = rng.integers(0, pos_idx.size, size=pos_idx.size)
            take_neg = rng.integers(0, neg_idx.size, size=neg_idx.size)
            idx = np.concatenate([pos_idx[take_pos], neg_idx[take_neg]])
        # Identical indices for both signals. This is the pairing.
        draws[r] = auroc(b[idx], y[idx]) - auroc(a[idx], y[idx])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:  # pragma: no cover - stratification prevents this
        return (delta, float("nan"), float("nan"), float("nan"), n_paired)

    alpha = (1.0 - level) / 2.0
    low = float(np.quantile(finite, alpha))
    high = float(np.quantile(finite, 1.0 - alpha))
    # Two-sided achieved significance level, centred on zero.
    if delta >= 0.0:
        tail = int(np.count_nonzero(finite <= 0.0))
    else:
        tail = int(np.count_nonzero(finite >= 0.0))
    p = min(1.0, 2.0 * (tail + 1) / (finite.size + 1))
    return (delta, low, high, p, n_paired)


def wilson_interval(k: int, n: int, *, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Closed form, no dependencies: with z the two-sided normal quantile,
    centre = (p + z^2/2n) / (1 + z^2/n) and half-width =
    z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n). Used for audit rates
    (clustering disagreements) where n is small and the Wald interval would
    collapse. Returns (NaN, NaN) for n == 0.
    """
    import math
    from statistics import NormalDist

    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0.0 < level < 1.0:
        raise ValueError(f"wilson_interval needs 0 < level < 1, got {level}")
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment. Returns adjusted p-values in input order.

    Holm rather than plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate, and the family here is 20 comparisons
    against one reference signal, which is enough for plain Bonferroni to hide a
    real effect.

    Adjusted values are made monotone non-decreasing along the sorted order,
    which is what makes "reject while adjusted p <= alpha" equivalent to the
    step-down procedure. NaN inputs are carried through as NaN and excluded from
    the count of comparisons, because a comparison that could not be computed is
    not a comparison that was made.
    """
    values = list(p_values)
    testable = [i for i, p in enumerate(values) if np.isfinite(p)]
    out = [float("nan")] * len(values)
    m = len(testable)
    if m == 0:
        return out
    order = sorted(testable, key=lambda i: values[i])
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = (m - rank) * values[i]
        running = max(running, adjusted)
        out[i] = float(min(1.0, running))
    return out
