"""TriviaQA builder, against a committed 12-row fixture.

The fixture is real rows from `rc.nocontext` validation with two edits: one
question blanked and one answer struct emptied of every alias, so both skip paths
run in CI without a download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unc_bench.datasets.triviaqa import TriviaQABuilder, _extract_aliases


class FixtureBuilder(TriviaQABuilder):
    """Reads the committed fixture instead of downloading."""

    def __init__(self, path: Path) -> None:
        super().__init__(path.parent)
        self._path = path

    @property
    def local_path(self) -> Path:
        return self._path


@pytest.fixture
def builder(fixtures_dir: Path) -> FixtureBuilder:
    return FixtureBuilder(fixtures_dir / "triviaqa_sample.parquet")


def test_loads_every_usable_row(builder: FixtureBuilder) -> None:
    # 12 rows, minus one blank question and one with no aliases at all.
    assert len(builder.load_candidates()) == 10


def test_qids_are_namespaced_and_unique(builder: FixtureBuilder) -> None:
    """The qid prefix is what keeps a mixed dataset's qids from colliding."""
    qids = [q.qid for q in builder.load_candidates()]
    assert all(q.startswith("triviaqa-") for q in qids)
    assert len(set(qids)) == len(qids)


def test_dataset_field_is_set(builder: FixtureBuilder) -> None:
    assert {q.dataset for q in builder.load_candidates()} == {"triviaqa"}


def test_canonical_value_is_the_first_alias(builder: FixtureBuilder) -> None:
    """A human-readable primary alias, matching the PopQA builder's convention."""
    first = builder.load_candidates()[0]
    assert first.gold_answers[0] == "David Seville"


def test_both_alias_lists_are_kept(builder: FixtureBuilder) -> None:
    """The dataset's own normalization is close to but not identical to ours.

    Keeping both means this project's `normalize_answer` runs over the raw and
    the pre-normalized forms, rather than trusting one upstream convention.
    """
    first = builder.load_candidates()[0]
    lowered = {a.casefold() for a in first.gold_answers}
    assert "david seville" in lowered
    assert "David Seville" in first.gold_answers


def test_sampling_is_deterministic(builder: FixtureBuilder) -> None:
    a = [q.qid for q in builder.build(4, seed=12345)]
    b = [q.qid for q in builder.build(4, seed=12345)]
    assert a == b


def test_a_different_seed_gives_a_different_sample(builder: FixtureBuilder) -> None:
    a = [q.qid for q in builder.build(4, seed=1)]
    b = [q.qid for q in builder.build(4, seed=2)]
    assert a != b


def test_asking_for_more_than_exists_raises(builder: FixtureBuilder) -> None:
    with pytest.raises(ValueError, match="only 10 are usable"):
        builder.build(50, seed=1)


def test_alias_extraction_handles_a_non_dict() -> None:
    """Parquet struct columns occasionally read back as None on a null row."""
    assert _extract_aliases(None) == ()
    assert _extract_aliases("not a struct") == ()


def test_alias_extraction_coerces_numpy_arrays() -> None:
    """The alias fields arrive as numpy arrays, not lists.

    A bare truthiness check on a numpy array raises "truth value of an array is
    ambiguous", which is an unhelpful failure to hit from inside a builder.
    """
    import numpy as np

    answer = {
        "value": "Cyrillic",
        "aliases": np.array(["Cyrillic letters"]),
        "normalized_aliases": np.array(["cyrillic"]),
    }
    assert _extract_aliases(answer) == ("Cyrillic", "Cyrillic letters", "cyrillic")


def test_alias_extraction_drops_duplicates_and_blanks() -> None:
    answer = {"value": "Rome", "aliases": ["Rome", "  ", "roma"], "normalized_aliases": ["rome"]}
    assert _extract_aliases(answer) == ("Rome", "roma", "rome")
