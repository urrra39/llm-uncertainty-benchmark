"""Paired level differences for the N-ablation, on synthetic columns.

Per-level CIs cannot test N=3 against N=5 (nested subsets of the same
samples), so the ablation stage records paired bootstrap differences. These
tests drive the pure helper with small resample counts: fast, no NLI model,
no checkpoints.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from unc_bench.stages.ablation import ablation_level_differences

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _labels(n_pos: int = 20, n_neg: int = 20) -> BoolArray:
    return np.array([True] * n_pos + [False] * n_neg, dtype=bool)


def test_identical_levels_have_zero_delta_and_high_p() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=40)
    columns = {1: {"b": scores.copy()}, 3: {"b": scores.copy()}}
    out = ablation_level_differences(columns, _labels(), [1, 3], resamples=200, seed=7, level=0.95)
    assert out["available"] is True
    assert out["reference_level"] == 3
    row = out["comparisons"][0]
    assert row["delta_vs_reference"] == 0.0
    assert row["p_value"] > 0.5
    assert row["p_value_holm"] > 0.5
    assert row["significant_holm"] is False
    assert row["n_paired"] == 40


def test_better_level_has_positive_delta_against_max() -> None:
    # Level 1 perfectly separates the classes; level 3 is noise. Against the
    # max level (3), level 1 must read as a positive improvement.
    y = _labels()
    good = np.array([3.0] * 20 + [0.0] * 20)
    noise = np.array([1.0, 0.0] * 20, dtype=np.float64)
    columns = {1: {"b": good}, 3: {"b": noise}}
    out = ablation_level_differences(columns, y, [1, 3], resamples=500, seed=11, level=0.95)
    row = next(c for c in out["comparisons"] if c["level"] == 1 and c["reference_level"] == 3)
    assert row["delta_vs_reference"] > 0.3
    assert row["ci_low"] > 0.0


def test_empty_levels_are_reported_not_raised() -> None:
    out = ablation_level_differences({}, _labels(), [], resamples=50, seed=0, level=0.95)
    assert out["available"] is False
