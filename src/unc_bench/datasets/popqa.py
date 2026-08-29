"""PopQA.

Entity-centric questions generated from Wikidata triples, with popularity
(monthly Wikipedia pageviews) attached to subject and object. The point of
including it is the long tail: most items are about entities almost nobody looks
up, so a small model gets a lot of them wrong, which is exactly what an
error-prediction benchmark needs.

Source is a single TSV, not a datasets-library loader. The `possible_answers`
column is a JSON-encoded list of aliases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from unc_bench.datasets.base import DatasetBuilder, cached_download, clean_alias_list
from unc_bench.types import Question

# The repo ships one file, test.tsv, holding all 14k items.
POPQA_URL = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv"


class PopQABuilder(DatasetBuilder):
    name = "popqa"

    @property
    def local_path(self) -> Path:
        return self.raw_dir / "popqa_test.tsv"

    def load_candidates(self) -> list[Question]:
        path = cached_download(POPQA_URL, self.local_path)
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

        required = {"id", "question", "possible_answers", "obj"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"popqa: source is missing columns {sorted(missing)}")

        out: list[Question] = []
        for row in frame.itertuples(index=False):
            question = str(row.question).strip()
            if not question:
                continue
            aliases = _parse_possible_answers(str(row.possible_answers))
            # `obj` is the canonical surface form; keep it first so the primary
            # alias is the one a human would write.
            canonical = str(row.obj).strip()
            if canonical:
                aliases = (canonical, *aliases)
            aliases = clean_alias_list(list(aliases))
            if not aliases:
                continue
            out.append(
                Question(
                    qid=f"popqa-{row.id}",
                    dataset=self.name,
                    question=question,
                    gold_answers=aliases,
                )
            )
        return out


def _parse_possible_answers(raw: str) -> tuple[str, ...]:
    """Parse the JSON-encoded alias list, tolerating malformed rows.

    A handful of rows have escaping that json cannot read. Those fall back to
    the canonical object alone rather than aborting the whole build.
    """
    text = raw.strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ()
    if isinstance(parsed, list):
        return tuple(str(x) for x in parsed)
    return (str(parsed),)
