"""Command-line entry point. Subcommands are wired up as stages land."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unc-bench", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.version:
        from unc_bench import __version__

        print(__version__)
        return 0
    print("no subcommand given; see --help", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
