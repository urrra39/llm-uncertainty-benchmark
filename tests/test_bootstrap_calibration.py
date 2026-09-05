"""Calibration of the paired bootstrap AUROC-difference test.

`t_random` sits 0.196 AUROC below the leader yet is not significantly worse
(p_holm = 0.0800). The README reads that as honesty; this module checks whether
it is instead an under-rejecting test. The estimator stacks four conservative
choices: a percentile achieved significance level not centred on the null,
doubling for two-sidedness, a (tail+1)/(n+1) correction, then Holm across 20.

Under a null where two signals are exchangeable draws, the rejection rate at
alpha=0.05 should be near 0.05. Under a planted alternative with a known gap,
the test should have power. Both are measured here with fixed seeds rather
than asserted.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from unc_bench.analysis.metrics import paired_bootstrap_auroc_diff

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _exchangeable_pair(
    rng: np.random.Generator, n_pos: int = 63, n_neg: int = 57
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Two signals drawn from the same distribution: the null is true."""
    base = rng.normal(size=n_pos + n_neg)
    reference = base + rng.normal(scale=0.1, size=n_pos + n_neg)
    other = base + rng.normal(scale=0.1, size=n_pos + n_neg)
    labels = np.array([True] * n_pos + [False] * n_neg, dtype=bool)
    return reference, other, labels


def _planted_gap(
    rng: np.random.Generator, gap: float = 0.196, n_pos: int = 63, n_neg: int = 57
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """A pair with a known AUROC gap of roughly `gap`, same class counts as run #2."""
    labels = np.array([True] * n_pos + [False] * n_neg, dtype=bool)
    # Reference separates weakly, other separates strongly; shift sets the gap.
    reference = np.concatenate([rng.normal(size=n_pos) + 0.3, rng.normal(size=n_neg)])
    other = np.concatenate([rng.normal(size=n_pos) + 0.3 + gap * 4.0, rng.normal(size=n_neg)])
    return reference, other, labels


def rejection_rate(
    trials: int, *, seed: int, resamples: int = 1000, alpha: float = 0.05, null: bool = True
) -> float:
    """Empirical rejection rate of the unadjusted paired p-value at `alpha`."""
    rng = np.random.default_rng(seed)
    rejects = 0
    for trial in range(trials):
        pair = _exchangeable_pair(rng) if null else _planted_gap(rng)
        _, _, _, p, _ = paired_bootstrap_auroc_diff(
            pair[0], pair[1], pair[2], resamples=resamples, seed=1000 + trial
        )
        rejects += p <= alpha
    return rejects / trials


def test_type_one_error_is_near_nominal() -> None:
    # 200 null trials at 1000 resamples: expect ~10 rejections; allow 3..20.
    # A test rejecting far below alpha is under-powered by construction.
    rate = rejection_rate(200, seed=4242)
    assert 0.015 <= rate <= 0.10, f"type-I rate {rate:.3f} is off nominal 0.05"


def test_planted_gap_has_power() -> None:
    # A 0.196 gap at run #2's class counts must usually reject unadjusted.
    rate = rejection_rate(50, seed=777, null=False)
    assert rate >= 0.5, f"power {rate:.3f} against a 0.196 gap is too low"
