"""Stage 1: draw the question set and write it to parquet.

Deterministic given `dataset_seed` and the source files. Each builder gets the
same seed, which is fine because they sample independent candidate pools; using
`seed + i` per builder would look more careful and change nothing.

A count of zero skips the builder entirely rather than constructing it. That
matters here: this session's configs ask for TriviaQA only, and instantiating the
PopQA builder would trigger a 14k-row TSV download for a sample of zero.
"""

from __future__ import annotations

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
    for name, count in wanted.items():
        if count <= 0:
            continue
        builder_cls = BUILDERS.get(name)
        if builder_cls is None:
            raise NotImplementedError(
                f"dataset_mix asks for {count} {name} questions but no builder is implemented"
            )
        builder = builder_cls(cfg.paths.raw_dir)
        drawn = builder.build(count, cfg.dataset_seed)
        print(f"[build_dataset] {name}: {len(drawn)} questions", flush=True)
        questions.extend(drawn)

    frame = questions_to_frame(questions)
    write_checkpoint(frame, paths.dataset)
    print(f"[build_dataset] wrote {len(frame)} rows to {paths.dataset}", flush=True)
    return int(len(frame))
