"""TriviaQA, `rc.nocontext` validation split.

Trivia-style questions written by humans, with a long alias list per answer. The
`nocontext` configuration strips the retrieved evidence documents, which is what
this benchmark wants: the question is whether the model knows the fact, not
whether it can read.

17,944 rows in the validation split. Read from the single parquet file the Hub
serves for this configuration rather than through the `datasets` library, which
would add a dependency and a second cache layer for one flat file. Same reasoning
as PopQA (docs/DECISIONS.md D10).

`answer.aliases` and `answer.normalized_aliases` are both kept. The normalized
list is lowercased and punctuation-stripped by the dataset authors, which is
close to but not identical to this project's `normalize_answer`, so both are
passed through and the project's own normalizer runs over all of them at match
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from unc_bench.datasets.base import DatasetBuilder, cached_download, clean_alias_list
from unc_bench.types import Question

# The Hub's parquet endpoint for this configuration returns exactly one shard.
TRIVIAQA_URL = (
    "https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/"
    "parquet/rc.nocontext/validation/0.parquet"
)


class TriviaQABuilder(DatasetBuilder):
    name = "triviaqa"

    @property
    def local_path(self) -> Path:
        return self.raw_dir / "triviaqa_rc_nocontext_validation.parquet"

    def load_candidates(self) -> list[Question]:
        path = cached_download(TRIVIAQA_URL, self.local_path)
        frame = pd.read_parquet(path, columns=["question", "question_id", "answer"])

        required = {"question", "question_id", "answer"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"triviaqa: source is missing columns {sorted(missing)}")

        questions = frame["question"].tolist()
        qids = frame["question_id"].tolist()
        answers = frame["answer"].tolist()

        out: list[Question] = []
        for question, qid, answer in zip(questions, qids, answers, strict=True):
            text = str(question).strip()
            if not text:
                continue
            aliases = _extract_aliases(answer)
            if not aliases:
                continue
            out.append(
                Question(
                    qid=f"triviaqa-{qid}",
                    dataset=self.name,
                    question=text,
                    gold_answers=aliases,
                )
            )
        return out


def _extract_aliases(answer: Any) -> tuple[str, ...]:
    """Pull the canonical value and every alias out of one answer struct.

    `value` goes first so the primary alias is the surface form a human would
    write, matching the PopQA builder's convention. Order matters only for
    display; matching tries every alias.

    The alias arrays arrive as numpy arrays inside the parquet struct, not lists,
    so they are coerced explicitly. A bare `if not aliases` on a numpy array
    raises "truth value of an array is ambiguous", which is a confusing failure
    to debug from inside a dataset builder.
    """
    if not isinstance(answer, dict):
        return ()
    parts: list[str] = []
    value = answer.get("value")
    if value is not None:
        parts.append(str(value))
    for key in ("aliases", "normalized_aliases"):
        raw = answer.get(key)
        if raw is None:
            continue
        parts.extend(str(a) for a in list(raw))
    return clean_alias_list(parts)
