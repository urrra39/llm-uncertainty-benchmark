"""PopQA builder, driven by a committed 12-row slice of the real TSV.

The fixture is real data, including one row whose `possible_answers` JSON I
deliberately truncated, so the malformed-row path is exercised in CI without a
download.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from unc_bench.datasets.base import (
    OfflineError,
    deduplicate_questions,
    gold_in_question,
    questions_to_frame,
)
from unc_bench.datasets.popqa import INVERSE_RELATIONS, PopQABuilder
from unc_bench.types import Question


@pytest.fixture
def builder(tmp_path: Path, fixtures_dir: Path) -> PopQABuilder:
    made = PopQABuilder(tmp_path)
    made.local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixtures_dir / "popqa_sample.tsv", made.local_path)
    return made


def test_loads_candidates_from_local_file(builder: PopQABuilder) -> None:
    got = builder.load_candidates()
    # 12 rows in, 12 out: the malformed row still keeps its canonical `obj`.
    assert len(got) == 12
    assert all(q.dataset == "popqa" for q in got)
    assert all(q.qid.startswith("popqa-") for q in got)


def test_canonical_object_is_the_first_alias(builder: PopQABuilder) -> None:
    by_qid = {q.qid: q for q in builder.load_candidates()}
    first = by_qid["popqa-4222362"]
    assert first.question == "What is George Rankin's occupation?"
    assert first.gold_answers[0] == "politician"
    assert "political leader" in first.gold_answers


def test_malformed_alias_json_falls_back_to_canonical(builder: PopQABuilder) -> None:
    # Row 12 has a truncated JSON list. It must survive with the `obj` alias
    # rather than aborting the build or arriving with no gold answer.
    got = [q for q in builder.load_candidates() if len(q.gold_answers) == 1]
    assert got, "expected the malformed row to yield exactly one alias"


def test_aliases_are_deduplicated(builder: PopQABuilder) -> None:
    for question in builder.load_candidates():
        assert len(set(question.gold_answers)) == len(question.gold_answers)
        assert all(a.strip() for a in question.gold_answers)


def test_sampling_is_deterministic_for_a_seed(builder: PopQABuilder) -> None:
    a = builder.build(5, seed=99)
    b = builder.build(5, seed=99)
    assert [q.qid for q in a] == [q.qid for q in b]


def test_different_seeds_give_different_samples(builder: PopQABuilder) -> None:
    a = [q.qid for q in builder.build(5, seed=1)]
    b = [q.qid for q in builder.build(5, seed=2)]
    assert a != b


def test_sample_order_is_stable_and_sorted_by_qid(builder: PopQABuilder) -> None:
    got = [q.qid for q in builder.build(6, seed=7)]
    assert got == sorted(got)


def test_asking_for_too_many_raises(builder: PopQABuilder) -> None:
    with pytest.raises(ValueError, match="only 12 are usable"):
        builder.build(500, seed=0)


def test_zero_requested_returns_empty(builder: PopQABuilder) -> None:
    assert builder.build(0, seed=0) == []


def test_offline_mode_refuses_to_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNC_BENCH_OFFLINE", "1")
    with pytest.raises(OfflineError, match="UNC_BENCH_OFFLINE=1"):
        PopQABuilder(tmp_path).load_candidates()


def test_frame_roundtrip_rejects_duplicate_qids(builder: PopQABuilder) -> None:
    questions = builder.build(3, seed=3)
    frame = questions_to_frame(questions)
    assert list(frame.columns) == ["qid", "dataset", "question", "gold_answers"]
    assert len(frame) == 3
    with pytest.raises(ValueError, match="duplicate qids"):
        questions_to_frame(questions + questions)


def test_inverse_relation_raises_without_the_escape_hatch(tmp_path: Path) -> None:
    assert "capital of" in INVERSE_RELATIONS
    with pytest.raises(ValueError, match="allow_inverse_relations"):
        PopQABuilder(tmp_path, relations=("capital of",))


def test_inverse_relation_loads_with_the_escape_hatch(tmp_path: Path) -> None:
    made = PopQABuilder(tmp_path, relations=("capital of",), allow_inverse_relations=True)
    assert made.relations == ("capital of",)


def test_gold_in_question_flags_the_echo_cases() -> None:
    assert gold_in_question("What is Rome the capital of?", ["Rome", "Italy"])
    assert gold_in_question("What is the capital of Nan?", ["Nan"])
    assert not gold_in_question("What is Seattle the capital of?", ["King County"])
    assert not gold_in_question("What color is Manchester United F.C.?", ["red"])
    # Substring is not enough: "nan" must not match inside "finance".
    assert not gold_in_question("Who runs finance?", ["Nan"])


def test_deduplicate_merges_alias_lists_and_keeps_first_qid() -> None:
    rows = [
        Question(
            qid="popqa-2",
            dataset="popqa",
            question="What is Rome the capital of?",
            gold_answers=("Lazio",),
        ),
        Question(
            qid="popqa-1",
            dataset="popqa",
            question="WHAT IS ROME THE CAPITAL OF? ",
            gold_answers=("Rome",),
        ),
    ]
    ordered = sorted(rows, key=lambda q: q.qid)
    merged, collapsed = deduplicate_questions(ordered)
    assert collapsed == 1
    assert len(merged) == 1
    assert merged[0].qid == "popqa-1"
    assert set(merged[0].gold_answers) == {"Rome", "Lazio"}


def test_frame_rejects_duplicate_question_text(builder: PopQABuilder) -> None:
    questions = builder.build(3, seed=3)
    clone = Question(
        qid="popqa-clone",
        dataset="popqa",
        question=questions[0].question.upper(),
        gold_answers=("Other",),
    )
    with pytest.raises(ValueError, match="duplicate question texts"):
        questions_to_frame([*questions, clone])
