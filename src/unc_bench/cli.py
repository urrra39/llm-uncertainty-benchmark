"""Command-line entry point: one subcommand per pipeline stage.

    build-dataset -> generate -> score-signals -> label -> analyze -> figures

Every stage takes `--config` and every stage is safe to rerun. `score-signals`
splits into two passes via `--family`, because the NLI model and the generator
cannot be resident at the same time in 2 GB (docs/DECISIONS.md D9): `actc` needs
no model, `b` loads DeBERTa, and `all` runs them in that order in one process
after the generator has already exited with its own stage.

Stage modules are imported inside the handlers, not at module scope. Some of
them pull in matplotlib or transformers, and `unc-bench --version` should not
pay for that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from unc_bench.config import Config


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="path to the YAML config (default: configs/default.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unc-bench", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    subparsers = parser.add_subparsers(dest="command")

    build = subparsers.add_parser("build-dataset", help="stage 1: draw the question set")
    _add_config(build)
    build.add_argument("--force", action="store_true", help="redraw even if a checkpoint exists")

    generate = subparsers.add_parser("generate", help="stage 2: greedy, samples, verification")
    _add_config(generate)
    generate.add_argument("--limit", type=int, default=None, help="stop after N new questions")

    score = subparsers.add_parser("score-signals", help="stage 3: signal columns")
    _add_config(score)
    score.add_argument(
        "--family",
        choices=("actc", "b", "all"),
        default="all",
        help="actc needs no model; b loads the NLI model; all runs actc then b",
    )
    score.add_argument("--limit", type=int, default=None, help="family B: stop after N rows")

    label = subparsers.add_parser("label", help="stage 4: exact match, judges, kappa")
    _add_config(label)

    analyze = subparsers.add_parser("analyze", help="stage 5: write results.json")
    _add_config(analyze)
    analyze.add_argument(
        "--no-figures", action="store_true", help="skip figure rendering after the report"
    )

    figures = subparsers.add_parser("figures", help="redraw figures from results.json alone")
    _add_config(figures)
    figures.add_argument("--view", default="primary", choices=("primary", "with_abstentions"))

    gate = subparsers.add_parser("pilot-gate", help="evaluate the pilot and recommend a mix")
    _add_config(gate)

    nondet = subparsers.add_parser("nondeterminism", help="rerun greedy prompts, report drift")
    _add_config(nondet)

    ablate = subparsers.add_parser("ablation", help="family B AUROC at N = 1, 2, 3, 5")
    _add_config(ablate)

    return parser


def _load(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.version:
        from unc_bench import __version__

        print(__version__)
        return 0

    command = getattr(args, "command", None)
    if command is None:
        print("no subcommand given; see --help", file=sys.stderr)
        return 2

    cfg = _load(args)

    if command == "build-dataset":
        from unc_bench.stages import build_dataset

        build_dataset.run(cfg, force=bool(args.force))
        return 0

    if command == "generate":
        from unc_bench.stages import generate

        generate.run(cfg, limit=args.limit)
        return 0

    if command == "score-signals":
        from unc_bench.stages import score_signals

        if args.family in ("actc", "all"):
            score_signals.run_actc(cfg)
        if args.family in ("b", "all"):
            score_signals.run_b(cfg, limit=args.limit)
        return 0

    if command == "label":
        from unc_bench.stages import label as label_stage

        label_stage.run(cfg)
        return 0

    if command == "analyze":
        from unc_bench.analysis.report import write_results

        write_results(cfg)
        if not args.no_figures:
            from unc_bench.analysis.figures import render_all

            render_all(cfg.paths.results_json, cfg.paths.figures_dir)
        return 0

    if command == "figures":
        from unc_bench.analysis.figures import render_all

        render_all(cfg.paths.results_json, cfg.paths.figures_dir, view=args.view)
        return 0

    if command == "pilot-gate":
        from unc_bench.stages.pilot_gate import evaluate

        payload = evaluate(cfg)
        # Exit 0 either way. The gate is advisory and its recommendation is
        # recorded in JSON; a non-zero exit would abort `make pilot` before the
        # measurement it exists to produce could be read.
        return 0 if payload["n_primary"] else 1

    if command == "nondeterminism":
        from unc_bench.stages.nondeterminism import run as nondet_run

        nondet_run(cfg)
        return 0

    if command == "ablation":
        from unc_bench.stages.ablation import run as ablation_run

        ablation_run(cfg)
        return 0

    print(f"unknown command {command!r}", file=sys.stderr)  # pragma: no cover
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
