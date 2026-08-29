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

    #: When set, keep only rows whose subject popularity is at or above this
    #: quantile of the whole file. 0.9 is the top popularity decile. Run #2 uses
    #: this to raise the base rate: the long tail is what made run #1 degenerate.
    popularity_quantile: float | None = None

    #: When set, keep only these Wikidata relation types (the `prop` column).
    #: Pilot iteration 1 measured 7.7% correct on the popularity-filtered slice,
    #: worse than TriviaQA's 11.1%, which showed that subject popularity is the
    #: wrong axis: `screenwriter of <famous film>` has a popular subject and an
    #: unrecallable object. Relation type is the axis that actually moves the
    #: base rate. See docs/DECISIONS.md, run #2 D1 pilot iteration 2.
    relations: tuple[str, ...] | None = None

    def __init__(
        self,
        raw_dir: Path,
        *,
        popularity_quantile: float | None = None,
        relations: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(raw_dir)
        self.popularity_quantile = popularity_quantile
        self.relations = relations

    def load_candidates(self) -> list[Question]:
        path = cached_download(POPQA_URL, self.local_path)
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

        required = {"id", "question", "possible_answers", "obj"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"popqa: source is missing columns {sorted(missing)}")

        frame = self._restrict_to_relations(frame)
        frame = self._restrict_to_popular(frame)

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

    def _restrict_to_relations(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only the requested relation types.

        PopQA's 16 relations differ enormously in how answerable they are for a
        small model. `capital`, `country` and `sport` are near-lookup facts with
        a small answer vocabulary. `screenwriter`, `producer` and `composer`
        require recalling one name out of thousands from film credits, and a
        0.5B model does not hold those. Filtering on relation is therefore a
        difficulty knob, and it is a much stronger one than subject popularity.

        An unknown relation name is an error rather than a silent empty slice,
        because a typo here would otherwise look like "the source has no such
        rows" and quietly shrink the dataset.
        """
        if self.relations is None:
            return frame
        if "prop" not in frame.columns:
            raise ValueError("popqa: relations was requested but the source has no prop column")
        available = set(frame["prop"].unique())
        unknown = sorted(set(self.relations) - available)
        if unknown:
            raise ValueError(
                f"popqa: unknown relation types {unknown}; available {sorted(available)}"
            )
        kept = frame.loc[frame["prop"].isin(self.relations)]
        print(
            f"[popqa] relations={list(self.relations)} keeps {len(kept)} of {len(frame)} rows",
            flush=True,
        )
        return kept

    def _restrict_to_popular(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only the most-viewed subjects, if a quantile was requested.

        `s_pop` is monthly Wikipedia pageviews for the question's subject entity.
        It is the field the PopQA authors provide for exactly this purpose, so no
        proxy is needed and none is used. Rows whose `s_pop` does not parse as a
        number are dropped rather than imputed: a missing popularity cannot be
        placed relative to a quantile, and guessing would silently mix tail
        entities back into a slice whose whole purpose is to exclude them.
        """
        if self.popularity_quantile is None:
            return frame
        if "s_pop" not in frame.columns:
            raise ValueError(
                "popqa: popularity_quantile was requested but the source has no s_pop column"
            )
        pop = pd.to_numeric(frame["s_pop"], errors="coerce")
        usable = frame.loc[pop.notna()].copy()
        usable["_pop"] = pop.loc[pop.notna()]
        cutoff = float(usable["_pop"].quantile(self.popularity_quantile))
        kept = usable.loc[usable["_pop"] >= cutoff].drop(columns=["_pop"])
        print(
            f"[popqa] popularity_quantile={self.popularity_quantile}: "
            f"s_pop cutoff {cutoff:.0f} keeps {len(kept)} of {len(frame)} rows",
            flush=True,
        )
        return kept


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
