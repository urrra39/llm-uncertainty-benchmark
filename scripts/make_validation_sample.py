"""Build a human-validation sample CSV for a run (E6).

Reads the run's labels + generations artifacts and writes 100 rows balanced
on machine label with an empty `human_label` column. Deterministic given the
config's split seed. When the minority class is short of 50, all of its rows
are taken and the majority fills to 100; the actual split is printed and
belongs in any prose citing the file. Nothing is labelled here — the human
column ships empty on purpose (docs/HUMAN_LABELING.md).

    uv run python scripts/make_validation_sample.py --config configs/run2b_clean.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import numpy as np
    import pandas as pd

    from unc_bench.config import Config
    from unc_bench.stages.common import StagePaths, json_load, read_checkpoint

    cfg = Config.load(args.config)
    paths = StagePaths.of(cfg)
    labels = read_checkpoint(paths.labels)
    generations = read_checkpoint(paths.generations)
    if labels is None or generations is None:
        print("labels or generations artifact absent; run label first", file=sys.stderr)
        return 2

    gen = generations.set_index("qid")
    pool: dict[str, list[dict[str, object]]] = {"correct": [], "incorrect": []}
    for record in labels.to_dict(orient="records"):
        label = str(record["label"])
        if label not in pool:
            continue
        qid = str(record["qid"])
        growth = gen.loc[qid] if qid in gen.index else None
        gold = ""
        if growth is not None:
            aliases = json_load(str(growth["gold_answers"])) or []
            gold = " | ".join(str(a) for a in aliases)
            question = str(growth["question"])
            dataset = str(growth["dataset"])
            answer = str(growth.get("greedy_answer") or "")
        else:
            question = dataset = answer = ""
        pool[label].append(
            {
                "qid": qid,
                "dataset": dataset,
                "question": question,
                "gold_answers": gold,
                "model_answer": answer,
                "heuristic_verdict": label,
                "judge_primary_verdict": str(record.get("judge_primary_verdict") or ""),
                "judge_secondary_verdict": str(record.get("judge_secondary_verdict") or ""),
                "machine_label": label,
                "machine_label_source": str(record.get("source") or ""),
                "human_label": "",
            }
        )

    rng = np.random.default_rng(cfg.split.seed)
    want_each = args.n // 2
    picked: list[dict[str, object]] = []
    for verdict in ("correct", "incorrect"):
        rows = pool[verdict]
        order = rng.permutation(len(rows)) if rows else []
        picked.extend(rows[int(i)] for i in order[:want_each])
    # Minority short of half: fill from the majority to reach n.
    majority = max(pool, key=lambda v: len(pool[v]))
    if len(picked) < args.n:
        have = {id(r) for r in picked}
        rest = [r for r in pool[majority] if id(r) not in have]
        picked.extend(rest[: args.n - len(picked)])
    frame = pd.DataFrame(picked).sort_values("qid").reset_index(drop=True)

    out = Path(cfg.paths.human_validation_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    counts = frame["machine_label"].value_counts().to_dict()
    print(f"wrote {len(frame)} rows to {out}: {counts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
