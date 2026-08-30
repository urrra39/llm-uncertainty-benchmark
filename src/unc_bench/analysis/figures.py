"""Figures, drawn from results.json alone.

Deliberately decoupled from the pipeline: `make figures` reads the JSON and
nothing else, so a plotting change never requires rerunning a model and the
figures cannot disagree with the numbers in the results file.

Six figures: the AUROC table with CIs, the risk-coverage curves, a reliability
diagram, the signal correlation heatmap, family-B AUROC against sample count,
and AUROC against measured cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend, selected before pyplot is imported. There is no
# display here and the default backend would fail on import.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from unc_bench.signals.base import FAMILY_LABELS

FAMILY_COLORS = {
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#d62728",
    "T": "#7f7f7f",
}
DPI = 150


class NothingToPlotError(RuntimeError):
    """A figure has no data behind it.

    Raised rather than drawing an empty axes, and caught by `render_all` so one
    undrawable figure does not abort the others. This happens for real: a view
    whose rows are all one class has no AUROC anywhere, and that is a fact about
    the run rather than a bug to crash on.
    """


def _num(value: Any, default: float = float("nan")) -> float:
    """Read a number that `results.json` may legitimately store as null.

    The report writes null wherever a value was not measurable, so every read
    here has to tolerate it. Defaults to NaN, which the plotting code already
    filters, rather than to 0.0, which would draw a point at a value nothing
    measured.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_results(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to an object")
    return data


def plot_auroc(results: dict[str, Any], out: Path, *, view: str = "primary") -> Path:
    """Every signal's AUROC with its bootstrap CI, grouped by family.

    The 0.5 line is drawn because it is the only reference that matters: a
    signal whose CI crosses it has not been shown to carry information at this n.
    """
    payload = results["views"][view]
    signals = payload["signals"]
    ordered = list(payload["ranking"])
    if not ordered:
        raise NothingToPlotError(
            f"view {view!r} has no signal with a defined AUROC "
            f"(n={payload['n']}, base rate {payload['base_rate_incorrect']})"
        )

    points = [_num(signals[n]["auroc"]["point"]) for n in ordered]
    # A CI that could not be estimated falls back to the point estimate, which
    # draws a bare marker with no error bar. That is the honest picture: the
    # AUROC is known and its uncertainty is not.
    lows = [_num(signals[n]["auroc"]["ci_low"], p) for n, p in zip(ordered, points, strict=True)]
    highs = [_num(signals[n]["auroc"]["ci_high"], p) for n, p in zip(ordered, points, strict=True)]
    colors = [FAMILY_COLORS.get(signals[n]["family"], "#333333") for n in ordered]

    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(ordered) + 2.0))
    y = np.arange(len(ordered))
    lower = [max(p - lo, 0.0) for p, lo in zip(points, lows, strict=True)]
    upper = [max(hi - p, 0.0) for p, hi in zip(points, highs, strict=True)]
    ax.errorbar(
        points,
        y,
        xerr=[lower, upper],
        fmt="o",
        markersize=5,
        capsize=3,
        linestyle="none",
        ecolor="#999999",
        elinewidth=1.2,
        zorder=3,
    )
    for yi, point, color in zip(y, points, colors, strict=True):
        ax.plot([point], [yi], "o", color=color, markersize=6, zorder=4)
    ax.axvline(0.5, color="#000000", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC for predicting an INCORRECT answer (higher is better)")
    n = payload["n"]
    base = _num(payload["base_rate_incorrect"])
    level = int(round(100 * results["analysis_config"]["ci_level"]))
    ax.set_title(
        f"{results['run_name']}  |  n={n}, base rate {base:.1%} incorrect\n"
        f"{level}% percentile bootstrap CI, "
        f"{results['analysis_config']['bootstrap_resamples']:,} resamples",
        fontsize=10,
    )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=FAMILY_COLORS[f], label=label)
        for f, label in FAMILY_LABELS.items()
        if f in FAMILY_COLORS
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_risk_coverage(
    results: dict[str, Any], out: Path, *, view: str = "primary", top_k: int = 5
) -> Path:
    """Risk-coverage curves for the best few signals, plus the random baseline.

    `t_random` is always included whether or not it ranks. It is the null: a flat
    line at the base rate, and any curve that does not sit below it is not buying
    anything.
    """
    payload = results["views"][view]
    signals = payload["signals"]
    chosen = list(payload["ranking"][:top_k])
    if "t_random" in signals and "t_random" not in chosen:
        chosen.append("t_random")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name in chosen:
        rc = signals[name]["risk_coverage"]
        if not rc["coverage"]:
            continue
        aurc = _num(signals[name]["aurc"])
        style = "--" if name == "t_random" else "-"
        ax.plot(
            rc["coverage"],
            rc["risk"],
            style,
            linewidth=1.6,
            color=FAMILY_COLORS.get(signals[name]["family"], "#333333"),
            label=f"{name}  (AURC {aurc:.3f})",
        )
    base = _num(payload["base_rate_incorrect"])
    ax.axhline(base, color="#000000", linestyle=":", linewidth=1.0, label=f"base rate {base:.3f}")
    ax.set_xlabel("coverage: fraction of questions answered, lowest-risk first")
    ax.set_ylabel("risk: error rate among answered questions")
    ax.set_title(
        f"Risk-coverage, {results['run_name']}  |  n={payload['n']}",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_calibration(
    results: dict[str, Any], out: Path, *, view: str = "primary", top_k: int = 3
) -> Path:
    """Reliability diagram for the top few signals after Platt scaling.

    Empty bins are simply absent from the line rather than interpolated across.
    At n=150 over 10 bins several bins hold nothing, and a smooth curve through
    them would be drawn entirely from data that does not exist.
    """
    payload = results["views"][view]
    signals = payload["signals"]
    chosen = list(payload["ranking"][:top_k])

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="perfect calibration")
    for name in chosen:
        cal = signals[name]["calibration"]
        conf = np.array(cal["bin_confidence"], dtype=np.float64)
        acc = np.array(cal["bin_accuracy"], dtype=np.float64)
        counts = np.array(cal["bin_counts"], dtype=np.int64)
        keep = (counts > 0) & np.isfinite(conf) & np.isfinite(acc)
        if not np.any(keep):
            continue
        ax.plot(
            conf[keep],
            acc[keep],
            "o-",
            markersize=4,
            linewidth=1.4,
            color=FAMILY_COLORS.get(signals[name]["family"], "#333333"),
            label=f"{name}  (ECE {_num(signals[name]['ece']):.3f})",
        )
    ax.set_xlabel("predicted P(incorrect), Platt-scaled on the train split")
    ax.set_ylabel("observed fraction incorrect")
    ax.set_title(
        f"Reliability, {results['run_name']}  |  "
        f"{results['analysis_config']['ece_bins']} bins, held-out rows only",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out)


def plot_correlation(results: dict[str, Any], out: Path, *, view: str = "primary") -> Path:
    """Spearman correlation heatmap over every signal.

    The question this answers: how much of the table is one measurement wearing
    twenty-one hats. Blocks of near-1.0 inside a family mean the family has one
    degree of freedom, and a high correlation between family A and
    `t_answer_length` is the specific confound family T exists to expose.
    """
    payload = results["views"][view]
    names = payload["correlation"]["names"]
    raw = payload["correlation"]["spearman"]
    matrix = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in raw], dtype=np.float64
    )

    fig, ax = plt.subplots(figsize=(9.0, 7.6))
    # NaN cells are passed through as-is: `imshow` routes non-finite values to
    # the colormap's "bad" colour, so setting that explicitly is enough and an
    # undefined correlation renders grey rather than as the midpoint colour,
    # which would read as rho = 0.
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(matrix, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(
        f"Spearman correlation between signals, {results['run_name']}  |  n={payload['n']}\n"
        "grey = undefined (constant column or fewer than three shared rows)",
        fontsize=10,
    )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman rho")
    fig.tight_layout()
    return _save(fig, out)


def plot_n_ablation(results: dict[str, Any], out: Path, *, view: str = "primary") -> Path:
    """AUROC against sample count N for every family-B signal (D6).

    The figure that decides whether family B's cost is justified. Each extra
    sample is another full generation, so a curve that flattens at N=2 or N=3
    means the 5-sample configuration is paying for nothing. Read from
    `ablation` in the results file, which the ablation stage recomputes from the
    first N of the same five samples at every level.
    """
    del view  # the ablation is computed on the primary view only
    ablation = results.get("ablation")
    if not ablation or not ablation.get("by_n"):
        raise NothingToPlotError("results.json carries no N-ablation; run the ablation stage")

    levels = [int(n) for n in ablation["levels"]]
    signals = list(ablation["signals"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for name in signals:
        ys = [_num(ablation["by_n"][str(n)]["signals"].get(name, {}).get("point")) for n in levels]
        ax.plot(levels, ys, marker="o", linewidth=1.6, markersize=5, label=name)

    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0)
    ax.annotate(
        "chance",
        xy=(levels[0], 0.5),
        xytext=(2, 4),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
    )
    ax.set_xlabel("samples used (N)")
    ax.set_ylabel("AUROC (positive class = incorrect)")
    ax.set_xticks(levels)
    ax.set_title(
        f"Self-consistency AUROC against sample count\n"
        f"n={ablation.get('n_rows')} rows, same five samples sliced at each N"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    return _save(fig, out)


def plot_cost_vs_auroc(results: dict[str, Any], out: Path, *, view: str = "primary") -> Path:
    """AUROC against measured cost, which is the study's actual question (D7).

    Cost on the x axis is the multiple of the single greedy answer every signal
    needs, so 1x means free and 6x means five extra generations plus the NLI
    pass. A signal is only interesting if nothing cheaper sits above and to the
    left of it, so the Pareto frontier is drawn explicitly rather than left for
    the reader to trace.
    """
    cost = results.get("cost")
    signals = ((results.get("views") or {}).get(view) or {}).get("signals") or {}
    if not cost or not cost.get("signals") or not signals:
        raise NothingToPlotError("results.json carries no cost table")

    points: list[tuple[float, float, str, str]] = []
    for name, entry in cost["signals"].items():
        auroc_value = _num(signals.get(name, {}).get("auroc", {}).get("point"))
        multiplier = _num(entry.get("cost_multiplier"))
        if np.isfinite(auroc_value) and np.isfinite(multiplier):
            points.append((multiplier, auroc_value, name, str(entry.get("family", "T"))))
    if not points:
        raise NothingToPlotError("no signal has both a cost and a defined AUROC")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for multiplier, auroc_value, name, family in points:
        ax.scatter(
            multiplier,
            auroc_value,
            s=42,
            color=FAMILY_COLORS.get(family, "#7f7f7f"),
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            name,
            xy=(multiplier, auroc_value),
            xytext=(5, 2),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
        )

    # Pareto frontier: cheapest-first, keep a point only if nothing cheaper
    # already scored at least as well.
    frontier: list[tuple[float, float]] = []
    best = -float("inf")
    for multiplier, auroc_value, _, _ in sorted(points, key=lambda p: (p[0], -p[1])):
        if auroc_value > best:
            best = auroc_value
            frontier.append((multiplier, auroc_value))
    if len(frontier) > 1:
        ax.step(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            where="post",
            color="#333333",
            linewidth=1.1,
            linestyle="--",
            zorder=2,
            label="Pareto frontier",
        )
        ax.legend(fontsize=8, loc="lower right")

    ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_xlabel("cost (multiple of one greedy answer)")
    ax.set_ylabel("AUROC (positive class = incorrect)")
    ax.set_title(
        "Discrimination against measured cost\n"
        + ", ".join(f"{key}={FAMILY_LABELS.get(key, key)}" for key in ("A", "B", "C", "T"))
    )
    ax.grid(alpha=0.3)
    return _save(fig, out)


def _save(fig: Any, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out}", flush=True)
    return out


def render_all(results_path: Path, figures_dir: Path, *, view: str = "primary") -> list[Path]:
    """Every figure, from the results file alone.

    One figure failing does not stop the others. A run whose primary view is
    degenerate still produces a usable correlation heatmap, and losing it to an
    exception raised by an unrelated plot would be pointless.
    """
    results = load_results(results_path)
    jobs = (
        ("auroc.png", plot_auroc),
        ("risk_coverage.png", plot_risk_coverage),
        ("calibration.png", plot_calibration),
        ("correlation.png", plot_correlation),
        ("n_ablation.png", plot_n_ablation),
        ("cost_vs_auroc.png", plot_cost_vs_auroc),
    )
    written: list[Path] = []
    for filename, plotter in jobs:
        try:
            written.append(plotter(results, figures_dir / filename, view=view))
        except NothingToPlotError as exc:
            print(f"[figures] skipped {filename}: {exc}", flush=True)
    return written
