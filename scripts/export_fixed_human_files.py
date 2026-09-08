"""Write the human-labelling targets for the FIXED label set (Part D prep).

The committed `data/human_validation_sample_run2b.csv` and
`data/fuzzy_decided_rows.csv` snapshot the PRE-FIX heuristic labels — they are
the round's evidence, and labelling them measures the 5.0% floor directly.
But run #2b's published label set is the FIXED one
(`data/run2b/labels_fixed.parquet`, analysed into
`results_run2b_fixedlabels.json`), and the human gate that authorises a
ranking must validate THAT set. This script writes the two fixed-label targets
(human_label empty, as always):

  data/human_validation_sample_run2b_fixed.csv — 100 rows balanced on the
    fixed machine label, with the same 11-column schema as the pre-fix sample
    (the two judge-verdict columns are present and empty: no judge ran under
    the heuristic labeler)
  data/fuzzy_decided_rows_fixed.csv — the 73 non-exact rows under the fixed
    rule, with the one rule-ambiguous row's fuzzy_verdict left blank
    (unresolved: a human must decide it, it is never coerced)

Both files are ordered highest-value first, so a partial labelling session
stops only after the rows that matter most: the rule-unresolved and
rule-changed rows at the very front, then the fuzzy-decided rows ahead of the
exact-match rows in the sample.

The owner labels these with `unc-bench label-human --run run2b`
(which resolves to the fixed fuzzy file) or `--target sample`. Never touches a
human_label cell that is already filled; writes none itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _aliases(raw: object) -> list[str]:
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    return [str(a) for a in decoded] if isinstance(decoded, list) else []


def _moved_qids(run_dir: Path) -> set[str]:
    """qids the fixed relabel changed or left unresolved (label_fix_rows.json).
    These rows are where the old rule demonstrably erred, so they carry the
    highest labelling value under the fixed set."""
    path = run_dir / "label_fix_rows.json"
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("qid")) for row in payload.get("changes", []) if row.get("qid")}


def _priority_sort(
    frame, moved: set[str], *, is_fuzzy_file: bool
):
    """Reordered frame: highest labelling value first.

    - fuzzy file: the rule-unresolved row (blank fuzzy_verdict) first, then
      rows the relabel moved, then the rest;
    - sample: fuzzy-decided rows ahead of exact-match rows, and within each,
      moved rows first.
    Deterministic within each group (qid order).
    """

    def key(i: int) -> tuple[int, int, int, str]:
        qid = str(frame.at[i, "qid"])
        if is_fuzzy_file:
            blank = str(frame.at[i, "fuzzy_verdict"]).strip() == ""
            return (0 if blank else 1 if qid in moved else 2, 0, 0, qid)
        fuzzy = str(frame.at[i, "machine_label_source"]).strip() == "heuristic_fuzzy"
        return (0 if fuzzy else 1, 0 if qid in moved else 1, 0, qid)

    order = sorted(range(len(frame)), key=key)
    return frame.iloc[order].reset_index(drop=True)


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    import pandas as pd

    from unc_bench.config import Config
    from unc_bench.normalize import clean_model_answer
    from unc_bench.stages.common import read_checkpoint

    cfg = Config.load("configs/run2b_clean.yaml")
    run_dir = REPO_ROOT / "data" / "run2b"
    labels = read_checkpoint(run_dir / "labels_fixed.parquet")
    generations = read_checkpoint(run_dir / "generations.parquet")
    assert labels is not None and generations is not None
    merged = generations.merge(labels, on="qid", how="inner", validate="one_to_one")
    moved = _moved_qids(run_dir)

    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        label = str(record["label"])
        gold = " | ".join(_aliases(record.get("gold_answers")))
        rows.append(
            {
                "qid": str(record["qid"]),
                "dataset": str(record["dataset"]),
                "question": str(record["question"]),
                "gold_answers": gold,
                "model_answer": clean_model_answer(str(record.get("greedy_text") or "")),
                "fuzzy_verdict": label if label != "ambiguous" else "",
                "machine_label_source": str(record["source"]),
                "human_label": "",
            }
        )
    frame = pd.DataFrame(rows)

    # fuzzy-decided rows = everything the exact-match stage did NOT settle,
    # matching the committed fuzzy file's population (73 rows in run #2b).
    exact_qids = set(labels.loc[labels["source"] == "exact_match", "qid"].astype(str))
    fuzzy = frame.loc[~frame["qid"].isin(exact_qids)].reset_index(drop=True)
    fuzzy = _priority_sort(fuzzy, moved, is_fuzzy_file=True)
    fuzzy_out = REPO_ROOT / "data" / "fuzzy_decided_rows_fixed.csv"
    fuzzy_cols = ["qid", "question", "gold_answers", "model_answer", "fuzzy_verdict", "human_label"]
    fuzzy[fuzzy_cols].to_csv(fuzzy_out, index=False)

    # 100-row validation sample balanced on the fixed machine label, drawn
    # from ALL scored rows (exact-match and heuristic alike), mirroring
    # scripts/make_validation_sample.py's rule and seed.
    pool: dict[str, list[dict[str, object]]] = {"correct": [], "incorrect": []}
    for row in frame.to_dict(orient="records"):
        verdict = str(row["fuzzy_verdict"])
        if verdict in pool:
            row["heuristic_verdict"] = verdict
            row["machine_label"] = verdict
            row["judge_primary_verdict"] = ""
            row["judge_secondary_verdict"] = ""
            pool[verdict].append(row)
    rng = np.random.default_rng(cfg.split.seed)
    picked: list[dict[str, object]] = []
    for verdict in ("correct", "incorrect"):
        entries = pool[verdict]
        order = rng.permutation(len(entries)) if entries else []
        picked.extend(entries[int(i)] for i in order[:50])
    majority = max(pool, key=lambda v: len(pool[v]))
    if len(picked) < 100:
        have = {id(r) for r in picked}
        rest = [r for r in pool[majority] if id(r) not in have]
        picked.extend(rest[: 100 - len(picked)])
    sample = pd.DataFrame(picked)
    sample = _priority_sort(sample, moved, is_fuzzy_file=False)
    sample_out = REPO_ROOT / "data" / "human_validation_sample_run2b_fixed.csv"
    sample[
        [
            "qid",
            "dataset",
            "question",
            "gold_answers",
            "model_answer",
            "heuristic_verdict",
            "judge_primary_verdict",
            "judge_secondary_verdict",
            "machine_label",
            "machine_label_source",
            "human_label",
        ]
    ].to_csv(sample_out, index=False)

    print(f"fuzzy-decided fixed rows: {len(fuzzy)} -> {fuzzy_out}")
    print(
        f"fixed validation sample: {len(sample)} "
        f"(machine {sample['machine_label'].value_counts().to_dict()}) -> {sample_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
