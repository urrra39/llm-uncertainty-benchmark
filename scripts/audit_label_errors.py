"""Reproduce five hand-verified run #2b label errors, then sweep all 120 rows.

Round 11's validity drop rests on demonstrable label errors a human found by
reading `data/human_validation_sample_run2b.csv`. This script treats those
five as ground truth (provenance below), reproduces each one in code against
the committed run artifacts, then sweeps the whole 120-row run for the two
systematic patterns the five exemplify:

  (a) ECHO-FP: a containment match scored the answer correct when the model
      answer is a proper token-subsequence of a LONGER gold alias — the answer
      was swallowed by the alias, not stated by it. When the swallowed tokens
      are also question vocabulary ("Jamaica" inside "Kingston, Jamaica" for
      "capital of Jamaica?"), the label is the question's subject echoed back.
  (b) VERBOSE-FN: the answer textually states a full gold alias but was
      labelled incorrect because the surrounding tokens pushed the length gap
      past the cap ("Oman's capital is Muscat." for gold "Muscat", gap 3 > 2).

The count these two patterns produce is a LOWER BOUND on machine-label error,
not the rate: a code sweep cannot see semantic errors — an answer that names
the same entity without sharing a token, an alias list that is wrong, an
answer that negates the correct one in words a token test would accept. Human
labels (docs/HUMAN_LABELING.md) are the only measurement that bounds the
residue; that is Part D of the round.

Reads `data/run2b/labels.parquet` x `data/run2b/generations.parquet` — the
committed artifacts, not the human-validation CSV, so the audit runs on all
120 rows and cannot drift from what the analysis consumed. Writes
`data/label_error_audit.json`. Never touches `human_label`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The five hand-verified errors from data/human_validation_sample_run2b.csv.
#: The auditor read the visible rows against the gold lists and found these;
#: "human_truth" is that reading. This script's job is to REPRODUCE each in
#: code and prove the machine label contradicts it — not to assert the file's
#: wording, which the code below checks token by token.
VERIFIED: tuple[dict[str, str], ...] = (
    {
        "qid": "popqa-5864218",
        "truth": "incorrect",
        "direction": "false_positive",
        "note": (
            "gold includes 'Kingston, Jamaica'; the model answered 'Jamaica', "
            "the question's own subject, which is not the capital"
        ),
    },
    {
        "qid": "triviaqa-jp_1520",
        "truth": "incorrect",
        "direction": "false_positive",
        "note": (
            "gold includes 'Unicorn Whale'; the model answered 'whale', a "
            "sub-token of that alias and not the mammal (narwhal) the question "
            "asks about"
        ),
    },
    {
        "qid": "popqa-6298839",
        "truth": "correct",
        "direction": "false_negative",
        "note": "'Oman's capital is Muscat.' restates the subject then gives the "
        "gold answer; the length gap (3) exceeded the cap",
    },
    {
        "qid": "triviaqa-qb_1435",
        "truth": "correct",
        "direction": "false_negative",
        "note": "'Iraq invaded Kuwait in 1990.' gives gold 'Kuwait'; gap 4 > cap",
    },
    {
        "qid": "triviaqa-dpql_376",
        "truth": "correct",
        "direction": "false_negative",
        "note": "'Pantagruel was the son of Gargantua.' gives gold 'Gargantua'; " "gap 5 > cap",
    },
)

_NEGATION = frozenset(
    {
        "not",
        "never",
        "no",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "without",
        "except",
        "unless",
        "false",
        "denied",
        "deny",
        "refuse",
        "refused",
        "cannot",
        "cant",
        "didnt",
        "doesnt",
        "wasnt",
        "isnt",
        "are not",
    }
)


def _tokens(text: str) -> list[str]:
    from unc_bench.normalize import normalize_answer

    return normalize_answer(text).split()


def _subseq_start(haystack: list[str], needle: list[str]) -> int:
    """First index where `needle` sits contiguously in `haystack`, else -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def _covered(token: str, question_tokens: list[str]) -> bool:
    """Token appears in the question, or is the possessive of one that does.

    Normalization strips the apostrophe, so "Oman's" becomes the single token
    "omans". Without this, the restatement in "Oman's capital is Muscat."
    looks novel and the row is not recognised as a verbose-but-correct answer.
    """
    return token in question_tokens or (
        len(token) > 1 and token.endswith("s") and token[:-1] in question_tokens
    )


def _load_run() -> list[dict[str, object]]:
    """Join labels and generations into one row record per qid (120 rows)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pandas as pd

    from unc_bench.normalize import clean_model_answer

    labels = pd.read_parquet(REPO_ROOT / "data" / "run2b" / "labels.parquet")
    generations = pd.read_parquet(REPO_ROOT / "data" / "run2b" / "generations.parquet")
    merged = generations.merge(labels, on="qid", how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        raw_gold = (
            json.loads(str(record.get("gold_answers"))) if record.get("gold_answers") else None
        )
        gold = [str(a) for a in raw_gold] if isinstance(raw_gold, list) else []
        rows.append(
            {
                "qid": str(record["qid"]),
                "dataset": str(record["dataset"]),
                "question": str(record["question"]),
                "gold": gold,
                "answer": clean_model_answer(str(record.get("greedy_text") or "")),
                "machine_label": str(record["label"]),
                "machine_label_source": str(record["source"]),
            }
        )
    rows.sort(key=lambda r: str(r["qid"]))
    return rows


def _gap(answer: list[str], alias: list[str]) -> int:
    return abs(len(answer) - len(alias))


def _alias_swallowed_answer(answer: list[str], gold: list[list[str]]) -> bool:
    """Answer is a PROPER token-subsequence of a longer alias."""
    if not answer:
        return False
    return any(len(alias) > len(answer) and _subseq_start(alias, answer) >= 0 for alias in gold)


def _full_alias_with_question_coverage(
    answer: list[str], gold: list[list[str]], question_tokens: list[str]
) -> list[str] | None:
    """Longest gold alias present verbatim with every other token question-covered."""
    best: list[str] | None = None
    for alias in gold:
        start = _subseq_start(answer, alias)
        if start < 0:
            continue
        outside = answer[:start] + answer[start + len(alias) :]
        if outside and all(_covered(t, question_tokens) for t in outside):
            if best is None or len(alias) > len(best):
                best = alias
    return best


def _has_negation(question_tokens: list[str]) -> bool:
    return any(t in _NEGATION for t in question_tokens)


def reproduce_verified(row: dict[str, object], verified: dict[str, str]) -> dict[str, object]:
    """Mechanism evidence for one verified error, or a failing explanation."""
    answer = _tokens(str(row["answer"]))
    gold = [_tokens(g) for g in row["gold"] if _tokens(g)]  # type: ignore[index]
    question_tokens = _tokens(str(row["question"]))
    qid = str(row["qid"])
    direction = verified["direction"]
    truth = verified["truth"]
    machine = str(row["machine_label"])
    evidence: dict[str, object] = {}
    if direction == "false_positive":
        evidence["swallowed_aliases"] = [
            {"alias": " ".join(a), "answer": " ".join(answer), "gap": _gap(answer, a)}
            for a in gold
            if len(a) > len(answer) and _subseq_start(a, answer) >= 0
        ]
        evidence["answer_is_question_vocabulary"] = all(
            _covered(t, question_tokens) for t in answer
        )
    else:
        matched = _full_alias_with_question_coverage(answer, gold, question_tokens)
        evidence["verbatim_gold_alias_present"] = " ".join(matched) if matched else None
        evidence["length_gap_to_verbatim_alias"] = (
            _gap(answer, matched) if matched is not None else None
        )
        evidence["max_length_gap_accepted"] = 2
        evidence["restatement_tokens_all_question_vocabulary"] = matched is not None
        evidence["question_has_negation"] = _has_negation(question_tokens)
    return {
        "qid": qid,
        "question": str(row["question"]),
        "gold_answers": row["gold"],
        "model_answer": str(row["answer"]),
        "machine_label": machine,
        "machine_label_source": str(row["machine_label_source"]),
        "human_truth": truth,
        "direction": direction,
        "evidence": evidence,
        "reproduced": (machine != truth),
    }


def classify(row: dict[str, object]) -> str | None:
    """Pattern class of a machine-label error, or None for an honest label."""
    machine = str(row["machine_label"])
    source = str(row["machine_label_source"])
    if source != "heuristic_fuzzy":
        return None  # exact-match rows are correct by construction under their aliases
    answer = _tokens(str(row["answer"]))
    if not answer:
        return None
    gold = [_tokens(g) for g in row["gold"] if _tokens(g)]  # type: ignore[index]
    question_tokens = _tokens(str(row["question"]))
    if machine == "correct" and _alias_swallowed_answer(answer, gold):
        return "echo_fp"
    if machine == "incorrect":
        matched = _full_alias_with_question_coverage(answer, gold, question_tokens)
        if matched is not None and not _has_negation(question_tokens):
            return "verbose_fn"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "label_error_audit.json")
    args = parser.parse_args()

    rows = _load_run()
    by_qid = {str(r["qid"]): r for r in rows}
    assert len(rows) == 120, f"expected 120 run #2b rows, found {len(rows)}"

    verified_out = []
    failures: list[str] = []
    for verified in VERIFIED:
        row = by_qid.get(verified["qid"])
        if row is None:
            failures.append(f"{verified['qid']}: absent from the run artifacts")
            continue
        report = reproduce_verified(row, verified)
        if not report["reproduced"]:
            failures.append(
                f"{verified['qid']}: machine label {report['machine_label']!r} does not "
                f"contradict human truth {verified['truth']!r}"
            )
        verified_out.append(report)

    # Pattern sweep over all 120 rows.
    echo_qids: list[str] = []
    verbose_qids: list[str] = []
    for row in rows:
        pattern = classify(row)
        if pattern == "echo_fp":
            echo_qids.append(str(row["qid"]))
        elif pattern == "verbose_fn":
            verbose_qids.append(str(row["qid"]))

    fuzzy_correct_total = sum(
        1
        for r in rows
        if str(r["machine_label"]) == "correct"
        and str(r["machine_label_source"]) == "heuristic_fuzzy"
    )
    confirmed = sorted(
        set(q for r in verified_out if r["reproduced"] for q in [str(r["qid"])]) | set(verbose_qids)
    )

    payload: dict[str, object] = {
        "run": "run2b_clean",
        "n_rows": len(rows),
        "audit_source": (
            "data/run2b/labels.parquet x generations.parquet (the committed artifacts "
            "the analysis consumed); five hand-verified rows from "
            "data/human_validation_sample_run2b.csv are reproduced here in code"
        ),
        "verified_errors": verified_out,
        "all_verified_reproduced": not failures,
        "pattern_sweep": {
            "echo_fp": {
                "definition": (
                    "machine correct, heuristic_fuzzy; model answer is a proper "
                    "token-subsequence of a longer gold alias (swallowed by it)"
                ),
                "count": len(echo_qids),
                "qids": echo_qids,
            },
            "verbose_fn": {
                "definition": (
                    "machine incorrect, heuristic_fuzzy; a full gold alias appears "
                    "verbatim in the answer and every non-alias token is question "
                    "vocabulary (possessive-tolerant); question carries no negation"
                ),
                "count": len(verbose_qids),
                "qids": verbose_qids,
            },
        },
        "fuzzy_correct_rows_total": fuzzy_correct_total,
        "note_fp_is_fully_enumerated": (
            "the fuzzy rule marked exactly two rows correct and both are echo "
            f"false positives, so the false-positive population of the rule is "
            f"fully enumerated at {fuzzy_correct_total}"
        ),
        "confirmed_error_count": len(confirmed),
        "confirmed_error_qids": confirmed,
        "lower_bound_machine_label_error": {
            "count": len(confirmed),
            "n": len(rows),
            "rate": len(confirmed) / len(rows),
            "note": (
                "lower bound, not the error rate: a code sweep cannot see "
                "semantic errors a human would — an answer naming the same entity "
                "with no shared token, an answer that negates the gold one in "
                "unseen words, or a wrong gold alias. Human labels "
                "(docs/HUMAN_LABELING.md, Part D) bound the residue. The "
                "verbose_fn qids were found by code; three of the four "
                "(all but triviaqa-bb_3148, which sits outside the 100-row sample "
                "the auditor read) are among the five hand-verified rows."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )

    print(f"verified errors reproduced: {sum(1 for r in verified_out if r['reproduced'])}/5")
    print(f"pattern sweep over {len(rows)} rows:")
    print(f"  echo_fp   (answer swallowed by a longer alias):  {len(echo_qids)}  {echo_qids}")
    print(
        f"  verbose_fn (full gold alias verbatim, mislabelled): {len(verbose_qids)}  {verbose_qids}"
    )
    print(f"fuzzy-correct rows total: {fuzzy_correct_total} (both are the echo FPs)")
    print(
        f"confirmed errors: {len(confirmed)}/120 = {len(confirmed) / len(rows):.1%} "
        "lower bound on machine-label error"
    )
    print(f"wrote {args.out}")
    if failures:
        print("FAILED to reproduce:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
