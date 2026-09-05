"""Diagnose the inverse-relation echo contamination (Part A1 evidence base).

Reads `data/human_validation_sample.csv` and reports, per row, whether the
question is an inverse-relation phrasing ("What is X the capital of?"), whether
the model answer echoes the question subject, whether the gold alias list
contains the subject surface form, and the assigned label with its source.

Writes `data/echo_contamination_report.json`. Reads committed artifacts only;
labels nothing, invents nothing. Run with dev dependencies only:

    uv run python scripts/diagnose_echo_contamination.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "human_validation_sample.csv"
REPORT_PATH = REPO_ROOT / "data" / "echo_contamination_report.json"

# "What is <subject> the capital of?" — the inverse template. Case-insensitive;
# the subject is whatever sits between "is" and the trailing relation phrase.
INVERSE_RE = re.compile(r"^\s*what\s+is\s+(.+?)\s+the\s+capital\s+of\s*\??\s*$", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Local normalizer (lowercase, strip punctuation, collapse whitespace).

    Deliberately NOT `unc_bench.normalize.normalize_answer`: this script must
    run even if `src/` is absent, and the echo test only needs case/punctuation
    robustness rather than the full SQuAD folding.
    """
    out = text.casefold()
    out = re.sub(r"[^\w\s]", " ", out, flags=re.UNICODE)
    return re.sub(r"\s+", " ", out).strip()


def _tokens(text: str) -> list[str]:
    return _normalize(text).split() if _normalize(text) else []


def _contains(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[start : start + len(needle)] == needle
        for start in range(len(haystack) - len(needle) + 1)
    )


def diagnose_row(row: dict[str, str]) -> dict[str, Any]:
    question = row.get("question", "")
    answer = row.get("model_answer", "")
    gold_raw = row.get("gold_answers", "")
    match = INVERSE_RE.match(question or "")
    subject = match.group(1).strip() if match else ""
    answer_tokens = _tokens(answer or "")
    subject_tokens = _tokens(subject)
    echo_exact = bool(subject_tokens) and answer_tokens == subject_tokens
    echo_contained = bool(subject_tokens) and (
        echo_exact
        or _contains(answer_tokens, subject_tokens)
        or _contains(subject_tokens, answer_tokens)
    )
    aliases = [a.strip() for a in (gold_raw or "").split("|")]
    subject_in_alias = False
    for alias in aliases:
        alias_tokens = _tokens(alias)
        if alias_tokens and (
            alias_tokens == subject_tokens
            or _contains(alias_tokens, subject_tokens)
            or _contains(subject_tokens, alias_tokens)
        ):
            subject_in_alias = True
            break
    return {
        "qid": row.get("qid", ""),
        "inverse_relation_phrasing": match is not None,
        "question_subject": subject,
        "model_answer": answer,
        "answer_echoes_subject_exact": echo_exact,
        "answer_echoes_subject": echo_contained,
        "gold_contains_subject": subject_in_alias,
        "machine_label": row.get("machine_label", ""),
        "machine_label_source": row.get("machine_label_source", ""),
    }


def main() -> int:
    if not CSV_PATH.exists():
        print(f"missing input: {CSV_PATH}", file=sys.stderr)
        return 2
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    diagnosed = [diagnose_row(row) for row in rows]
    inverse = [d for d in diagnosed if d["inverse_relation_phrasing"]]
    echo = [d for d in inverse if d["answer_echoes_subject"]]
    echo_exact = [d for d in inverse if d["answer_echoes_subject_exact"]]
    echo_correct = [d for d in echo if d["machine_label"] == "correct"]
    echo_incorrect = [d for d in echo if d["machine_label"] == "incorrect"]
    # Rows whose label is decided purely by subject-in-alias-list: echo rows
    # where (subject in alias list) == (label correct). For exact-match rows
    # this is definitional (answer == subject); for judged rows it records
    # that the judge agreed with the alias happenstance.
    decided_by_alias = [
        d["qid"] for d in echo if d["gold_contains_subject"] == (d["machine_label"] == "correct")
    ]
    report = {
        "source": "data/human_validation_sample.csv",
        "n_rows": len(diagnosed),
        "n_inverse_phrased": len(inverse),
        "n_echo": len(echo),
        "n_echo_exact": len(echo_exact),
        "n_echo_labelled_correct": len(echo_correct),
        "n_echo_labelled_incorrect": len(echo_incorrect),
        "n_echo_decided_by_subject_in_alias_list": len(decided_by_alias),
        "echo_qids_decided_by_subject_in_alias_list": sorted(decided_by_alias),
        "contingency_echo_vs_correct": {
            "echo_and_correct": len(echo_correct),
            "echo_and_incorrect": len(echo_incorrect),
        },
        "note": (
            "Same model failure (echoing the subject) receives opposite labels "
            "depending on alias-list happenstance. Near-random labels injected "
            "into the confident stratum drive AUROC toward 0.50."
        ),
        "rows": diagnosed,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"rows={len(diagnosed)} inverse={len(inverse)} echo={len(echo)} "
        f"(exact {len(echo_exact)}) correct={len(echo_correct)} "
        f"incorrect={len(echo_incorrect)} decided_by_alias={len(decided_by_alias)}",
        flush=True,
    )
    print(f"wrote {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
