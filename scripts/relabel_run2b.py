"""Re-label run #2b under the fixed no-judge rule (A3, round 11).

The rule in `unc_bench.labeling.fuzzy_rule` replaces the containment rule that
produced the round's demonstrable label errors. This script recomputes the
label for all 120 run #2b rows from the committed generations (question, gold
aliases, greedy answer) and writes the result as `data/run2b/labels_fixed.parquet`
alongside the original heuristic `labels.parquet`, so BOTH label sets stay
committed and the relabel is a measurement, not an overwrite.

Rows the original exact-match stage settled keep their exact-match label;
heuristic rows are re-decided by the fixed rule, where the rule's AMBIGUOUS
verdict is stored as label `ambiguous` with source `heuristic_unresolved`
(queued for a human, never coerced). Judge columns stay empty — no judge was
asked under either label set.

Writes:
  data/run2b/labels_fixed.parquet   — the fixed label set, same schema
  data/run2b/label_fix_rows.json    — per-row old -> new for every change
Prints the class counts under both sets. Reads only committed artifacts;
never touches `human_label`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Columns of the committed labels.parquet; the fixed set keeps the same schema.
LABEL_COLUMNS = [
    "qid",
    "label",
    "source",
    "judge_raw",
    "judge_primary_verdict",
    "judge_secondary_verdict",
]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import pandas as pd

    from unc_bench.labeling import fuzzy_rule
    from unc_bench.normalize import clean_model_answer

    run_dir = REPO_ROOT / "data" / "run2b"
    labels = pd.read_parquet(run_dir / "labels.parquet")
    generations = pd.read_parquet(run_dir / "generations.parquet")
    merged = generations.merge(labels, on="qid", how="inner", validate="one_to_one")

    fixed_rows: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    counts: dict[str, int] = {"correct": 0, "incorrect": 0, "ambiguous": 0}
    sources: dict[str, int] = {"exact_match": 0, "heuristic_fuzzy": 0, "heuristic_unresolved": 0}
    for record in merged.to_dict(orient="records"):
        qid = str(record["qid"])
        machine = str(record["label"])
        source = str(record["source"])
        if source == "exact_match":
            label, new_source = machine, source
        else:
            raw_gold = json.loads(str(record["gold_answers"])) if record["gold_answers"] else None
            gold = [str(a) for a in raw_gold] if isinstance(raw_gold, list) else []
            answer = clean_model_answer(str(record.get("greedy_text") or ""))
            verdict = fuzzy_rule(answer, gold, str(record["question"]))
            label, new_source = (
                verdict,
                ("heuristic_unresolved" if verdict == "ambiguous" else "heuristic_fuzzy"),
            )
        if label != machine or new_source != source:
            changes.append(
                {
                    "qid": qid,
                    "old_label": machine,
                    "old_source": source,
                    "new_label": label,
                    "new_source": new_source,
                }
            )
        counts[label] += 1
        sources[new_source] += 1
        fixed_rows.append(
            {
                "qid": qid,
                "label": label,
                "source": new_source,
                "judge_raw": "",
                "judge_primary_verdict": "",
                "judge_secondary_verdict": "",
            }
        )

    fixed = (
        pd.DataFrame(fixed_rows, columns=LABEL_COLUMNS).sort_values("qid").reset_index(drop=True)
    )
    fixed.to_parquet(run_dir / "labels_fixed.parquet", index=False)
    (run_dir / "label_fix_rows.json").write_text(
        json.dumps(
            {
                "run": "run2b_clean",
                "rule": "unc_bench.labeling.fuzzy_rule (round 11 fixed no-judge rule)",
                "n_rows": len(fixed),
                "old_counts": {str(k): int(v) for k, v in labels["label"].value_counts().items()},
                "old_sources": {str(k): int(v) for k, v in labels["source"].value_counts().items()},
                "new_counts": counts,
                "new_sources": sources,
                "n_changed": len(changes),
                "changes": sorted(changes, key=lambda c: c["qid"]),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"old counts: {dict(labels['label'].value_counts())}")
    print(f"new counts: {counts}")
    print(f"changed rows: {len(changes)}")
    for change in sorted(changes, key=lambda c: c["qid"]):
        print(
            f"  {change['qid']:20s} {change['old_label']:9s} -> "
            f"{change['new_label']:9s} ({change['new_source']})"
        )
    print(f"wrote {run_dir / 'labels_fixed.parquet'}")
    print(f"wrote {run_dir / 'label_fix_rows.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
