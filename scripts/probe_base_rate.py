"""Measure the base rate of a question mix using greedy answers only.

Why this exists. The pilot gate in `configs/*_pilot.yaml` is a question about
one number: what fraction of answers are correct. Getting that number from the
`generate` stage costs about seven model calls per question (greedy, five
samples, two verification passes, one verbalized confidence) and measured 21.1
s/question on this machine. The base rate only needs the first of those calls.
This script runs that one call and nothing else, so a difficulty knob can be
tested for roughly a seventh of the cost of a full pilot.

It deliberately does not compute signals, does not write a checkpoint the real
stages would read, and does not call a judge. Its output is a base rate and a
per-dataset breakdown, printed and written to JSON. Labels come from normalized
exact match plus the fuzzy containment rule, which is the same heuristic path
the `label` stage falls back to. That understates correctness slightly relative
to a judge, so a mix that clears the gate here clears it there too.

Rows land in the same content-addressed response cache the real run uses, so
the greedy call paid for here is not paid for again during the full run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from unc_bench.cache import ResponseCache
from unc_bench.client import build_client
from unc_bench.config import Config
from unc_bench.datasets.base import frame_to_questions
from unc_bench.labeling import fuzzy_correct
from unc_bench.normalize import clean_model_answer, exact_match, is_abstention
from unc_bench.stages import build_dataset
from unc_bench.stages.generate import render_answer_prompt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="probe only the first N questions")
    parser.add_argument("--out", type=Path, default=None, help="write the summary here as JSON")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    build_dataset.run(cfg)

    import pandas as pd

    frame = pd.read_parquet(cfg.paths.artifacts_dir / "dataset.parquet")
    questions = frame_to_questions(frame)
    cfg.few_shot.assert_disjoint_from([q.question for q in questions])
    if args.limit is not None:
        questions = questions[: args.limit]

    cache = ResponseCache(cfg.paths.cache_dir)
    client = build_client(cfg.model_under_test, cache)

    per_dataset: dict[str, Counter[str]] = {}
    rows: list[dict[str, object]] = []
    for i, question in enumerate(questions, start=1):
        prompt = render_answer_prompt(client, cfg, question)
        gens = client.generate(
            prompt,
            max_new_tokens=cfg.model_under_test.max_new_tokens,
            temperature=cfg.greedy.temperature,
            top_p=cfg.greedy.top_p,
            seed=cfg.greedy.seed,
            top_logprobs=0,
        )
        raw = gens[0].text
        answer = clean_model_answer(raw, abstain_token=cfg.prompts.abstain_token)
        if is_abstention(raw, cfg.prompts.abstain_token) or not answer:
            outcome = "abstain"
        elif exact_match(answer, question.gold_answers) or fuzzy_correct(
            answer, question.gold_answers
        ):
            outcome = "correct"
        else:
            outcome = "incorrect"

        per_dataset.setdefault(question.dataset, Counter())[outcome] += 1
        rows.append(
            {
                "qid": question.qid,
                "dataset": question.dataset,
                "question": question.question,
                "answer": answer,
                "gold": list(question.gold_answers)[:3],
                "outcome": outcome,
            }
        )
        print(
            f"[{i}/{len(questions)}] {outcome:9s} {answer[:40]!r} <- {question.gold_answers[0]!r}",
            flush=True,
        )

    total: Counter[str] = Counter()
    for counts in per_dataset.values():
        total.update(counts)
    n = sum(total.values())
    summary = {
        "config": str(args.config),
        "n": n,
        "counts": dict(total),
        "correct_rate": total["correct"] / n if n else 0.0,
        "abstain_rate": total["abstain"] / n if n else 0.0,
        "per_dataset": {
            name: {
                "n": sum(counts.values()),
                "counts": dict(counts),
                "correct_rate": counts["correct"] / sum(counts.values()),
            }
            for name, counts in sorted(per_dataset.items())
        },
    }
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
