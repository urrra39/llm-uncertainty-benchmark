"""Stage 1: draw the question set and write it to parquet.

Deterministic given `dataset_seed` and the source files. Each builder gets the
same seed, which is fine because they sample independent candidate pools; using
`seed + i` per builder would look more careful and change nothing.

A count of zero skips the builder entirely rather than constructing it. That
matters here: this session's configs ask for TriviaQA only, and instantiating the
PopQA builder would trigger a 14k-row TSV download for a sample of zero.
"""

from __future__ import annotations

import json
from typing import Any

from unc_bench.config import Config
from unc_bench.datasets.base import DatasetBuilder, questions_to_frame
from unc_bench.datasets.popqa import PopQABuilder
from unc_bench.datasets.triviaqa import TriviaQABuilder
from unc_bench.stages.common import StagePaths, read_checkpoint, write_checkpoint
from unc_bench.types import Question

BUILDERS: dict[str, type[DatasetBuilder]] = {
    "popqa": PopQABuilder,
    "triviaqa": TriviaQABuilder,
}


def _construct(builder_cls: type[DatasetBuilder], name: str, cfg: Config) -> DatasetBuilder:
    """Build one dataset builder, passing the run's difficulty knobs.

    Run #2 raises the base rate by restricting both sources to their easier end
    (docs/DECISIONS.md, run #2 D1). The knobs live in the config so a rerun with
    `easy_slice: false` reproduces run #1's harder question distribution without
    a code change.
    """
    if name == "triviaqa":
        return TriviaQABuilder(
            cfg.paths.raw_dir,
            easy_only=cfg.difficulty.triviaqa_easy_only,
            min_aliases=cfg.difficulty.triviaqa_min_aliases,
            drop_gold_in_question=cfg.difficulty.drop_gold_in_question,
        )
    if name == "popqa":
        return PopQABuilder(
            cfg.paths.raw_dir,
            popularity_quantile=cfg.difficulty.popqa_popularity_quantile,
            relations=cfg.difficulty.popqa_relations,
            allow_inverse_relations=cfg.difficulty.allow_inverse_relations,
            drop_gold_in_question=cfg.difficulty.drop_gold_in_question,
        )
    return builder_cls(cfg.paths.raw_dir)


def run(cfg: Config, *, force: bool = False) -> int:
    """Build the dataset parquet. Returns the number of questions."""
    paths = StagePaths.of(cfg)
    if not force:
        existing = read_checkpoint(paths.dataset)
        if existing is not None and len(existing) == cfg.dataset_mix.total:
            print(
                f"[build_dataset] {paths.dataset} already holds "
                f"{len(existing)} questions; nothing to do",
                flush=True,
            )
            return int(len(existing))

    wanted = {
        "popqa": cfg.dataset_mix.popqa,
        "triviaqa": cfg.dataset_mix.triviaqa,
        "simpleqa": cfg.dataset_mix.simpleqa,
    }
    questions: list[Question] = []
    meta: dict[str, dict[str, Any]] = {}
    for name, count in wanted.items():
        if count <= 0:
            continue
        builder_cls = BUILDERS.get(name)
        if builder_cls is None:
            raise NotImplementedError(
                f"dataset_mix asks for {count} {name} questions but no builder is implemented"
            )
        builder = _construct(builder_cls, name, cfg)
        drawn = builder.build(count, cfg.dataset_seed)
        print(f"[build_dataset] {name}: {len(drawn)} questions", flush=True)
        questions.extend(drawn)
        pool = int(getattr(builder, "last_pool_unique", 0))
        margin = (pool / count) if count > 0 else float("nan")
        if count > 0 and pool and margin < 2.0:
            print(
                f"[build_dataset] WARNING {name}: sampling margin {margin:.2f}x "
                f"({pool} unique for {count} draws) is below 2x; the seed is "
                "near-decorative and the subset is a census, not a sample",
                flush=True,
            )
        meta[name] = {
            "drawn": len(drawn),
            "pool_unique": pool,
            "sampling_margin": margin,
            "dedup_collapsed": int(getattr(builder, "last_dedup_collapsed", 0)),
            "gold_leakage_dropped": int(getattr(builder, "last_gold_leakage_dropped", 0)),
            "gold_leakage_inspected": int(getattr(builder, "last_gold_leakage_inspected", 0)),
        }

    frame = questions_to_frame(questions)
    write_checkpoint(frame, paths.dataset)
    meta_path = cfg.paths.artifacts_dir / "dataset_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build_dataset] wrote {len(frame)} rows to {paths.dataset}", flush=True)
    return int(len(frame))
