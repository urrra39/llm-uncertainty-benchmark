"""Gold-list quality per PopQA relation, without running any model (Part C2).

Relation selection for run #3 must rest on alias-list completeness, not on
recall difficulty: `occupation` has genuinely multi-valued gold (politician
AND lawyer) and `place of birth` mixes city/country granularity, and both are
alias-completeness failures waiting to happen. For every relation this reports,
from the source file alone:

- candidates, mean/median alias-list length, single-alias fraction
  (high values are dangerous: one missing alias flips the label);
- gold-in-question fraction (echo scores correct);
- granularity-span fraction (one alias a token-subsequence of another, or
  aliases differing by more than two tokens — city-vs-country shape).

Writes `data/gold_quality_report.json`. Run with dev dependencies only:

    uv run python scripts/diagnose_gold_quality.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "data" / "gold_quality_report.json"


def _normalize(text: str) -> str:
    import re

    out = text.casefold()
    out = re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)
    return re.sub(r"\s+", " ", out).strip()


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)
    return normalized.split() if normalized else []


def _spans_granularity(aliases: list[list[str]]) -> bool:
    """True when the alias list mixes specificity levels.

    Heuristic, stated as one: one alias is a contiguous token-subsequence of
    another ("Delhi" inside "Delhi Sultanate"), or two aliases differ in token
    count by more than two ("Kostenets" vs a three-token district name).
    """
    for i, left in enumerate(aliases):
        for right in aliases[i + 1 :]:
            if not left or not right:
                continue
            if abs(len(left) - len(right)) > 2:
                return True
            short, long = sorted((left, right), key=len)
            if any(
                long[start : start + len(short)] == short
                for start in range(len(long) - len(short) + 1)
            ):
                return True
    return False


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from unc_bench.datasets.base import cached_download, gold_in_question
    from unc_bench.datasets.popqa import POPQA_URL, clean_alias_list

    frame = pd.read_csv(
        cached_download(POPQA_URL, REPO_ROOT / "data" / "raw" / "popqa_test.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    relations: dict[str, Any] = {}
    for prop in sorted(frame["prop"].unique()):
        group = frame[frame["prop"] == prop]
        alias_lists: list[list[str]] = []
        leaks = 0
        for row in group.itertuples(index=False):
            raw: Any = row.possible_answers
            parsed: list[str] = []
            try:
                decoded = json.loads(str(raw))
                parsed = [str(x) for x in decoded] if isinstance(decoded, list) else [str(decoded)]
            except json.JSONDecodeError:
                pass
            canonical = str(row.obj).strip()
            aliases = clean_alias_list(([canonical] if canonical else []) + parsed)
            alias_lists.append([_tokens(a) for a in aliases])
            if gold_in_question(str(row.question), list(aliases)):
                leaks += 1
        lengths = [len(a) for a in alias_lists]
        single = sum(1 for n in lengths if n == 1)
        granular = sum(1 for a in alias_lists if _spans_granularity(a))
        relations[prop] = {
            "candidates": int(len(group)),
            "mean_aliases": round(statistics.fmean(lengths), 2),
            "median_aliases": float(statistics.median(lengths)),
            "single_alias_fraction": round(single / len(lengths), 3),
            "gold_in_question_fraction": round(leaks / len(group), 3),
            "granularity_span_fraction": round(granular / len(group), 3),
            "example_question": str(group["question"].iloc[0]),
        }
    report = {
        "source": "PopQA test.tsv via the project's own builders",
        "selection_rule": (
            "prefer low single-alias fraction (alias omissions flip labels), "
            "low gold-in-question fraction (echo scores correct), low "
            "granularity-span fraction (city-vs-country ambiguity)"
        ),
        "relations": relations,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {REPORT_PATH} ({len(relations)} relations)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
