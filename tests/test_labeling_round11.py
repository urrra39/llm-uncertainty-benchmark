"""Round 11: the fixed no-judge rule relabels exactly the six demonstrable
run #2b errors, and the committed artifacts still carry the pre-fix set.

The human hand-check that dropped the round's validity score found five label
errors in data/human_validation_sample_run2b.csv; a code sweep
(`scripts/audit_label_errors.py`) reproduced those five and found a sixth
(triviaqa-bb_3148) in the 20 rows outside the 100-row sample. This file pins
the rule (`unc_bench.labeling.fuzzy_rule`) to that ground truth on the actual
committed rows, and pins the committed artifacts to their pre-fix state so a
relabel can only ever land as an ADDED label set, never a silent overwrite.

Data-coupled by design: these rows are the round's evidence, the same way
tests/test_fuzzy_label_length_bias.py couples to data/run2b/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RUN2B = REPO / "data" / "run2b"

#: The six rows whose machine label the round proves wrong, and the verdict the
#: fixed rule must give them (five hand-verified; bb_3148 found by the code
#: sweep, same textual structure as the verified verbatim-correct cases).
EXPECTED_FIXED: dict[str, str] = {
    "popqa-5864218": "incorrect",  # echo false positive: subject answered
    "triviaqa-jp_1520": "ambiguous",  # partial-alias echo; a human must decide
    "popqa-6298839": "correct",  # verbose-but-correct, restated then answered
    "triviaqa-qb_1435": "correct",  # verbose-but-correct
    "triviaqa-dpql_376": "correct",  # verbose-but-correct
    "triviaqa-bb_3148": "correct",  # verbose-but-correct (code-found)
}

#: The pre-fix machine labels of the same rows, as committed. This is what the
#: sweep counts as error; if a relabel ever overwrote the committed set, this
#: anchor moves and this test fails loudly rather than silently.
PRE_FIX_LABELS: dict[str, str] = {
    "popqa-5864218": "correct",
    "triviaqa-jp_1520": "correct",
    "popqa-6298839": "incorrect",
    "triviaqa-qb_1435": "incorrect",
    "triviaqa-dpql_376": "incorrect",
    "triviaqa-bb_3148": "incorrect",
}


def _rows() -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "src"))
    generations = pd.read_parquet(RUN2B / "generations.parquet")
    labels = pd.read_parquet(RUN2B / "labels.parquet")
    return generations.merge(labels, on="qid", how="inner", validate="one_to_one")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _rows()


def _gold_of(raw: object) -> list[str]:
    try:
        decoded = json.loads(str(raw))
    except (TypeError, ValueError):
        return []
    return [str(a) for a in decoded] if isinstance(decoded, list) else []


def test_fixed_rule_gives_the_verified_verdicts_on_the_actual_rows(frame: pd.DataFrame) -> None:
    from unc_bench.labeling import fuzzy_rule
    from unc_bench.normalize import clean_model_answer

    seen: set[str] = set()
    for record in frame.to_dict(orient="records"):
        qid = str(record["qid"])
        if qid not in EXPECTED_FIXED:
            continue
        seen.add(qid)
        answer = clean_model_answer(str(record.get("greedy_text") or ""))
        verdict = fuzzy_rule(answer, _gold_of(record.get("gold_answers")), str(record["question"]))
        assert verdict == EXPECTED_FIXED[qid], (
            f"{qid}: fixed rule gave {verdict!r}, expected {EXPECTED_FIXED[qid]!r} "
            f"for answer {answer!r}"
        )
    assert seen == set(
        EXPECTED_FIXED
    ), f"missing rows from run artifacts: {set(EXPECTED_FIXED) - seen}"


def test_relabel_changes_exactly_these_six_rows(frame: pd.DataFrame) -> None:
    from unc_bench.labeling import fuzzy_rule
    from unc_bench.normalize import clean_model_answer

    changed: dict[str, tuple[str, str]] = {}
    for record in frame.to_dict(orient="records"):
        qid = str(record["qid"])
        machine = str(record["label"])
        if str(record["source"]) == "exact_match":
            continue  # exact rows never pass through the no-judge rule
        answer = clean_model_answer(str(record.get("greedy_text") or ""))
        verdict = fuzzy_rule(answer, _gold_of(record.get("gold_answers")), str(record["question"]))
        if verdict != machine:
            changed[qid] = (machine, verdict)
    assert set(changed) == set(EXPECTED_FIXED), (
        "the fixed rule must move exactly the six demonstrable errors, no more: "
        f"{sorted(set(changed) ^ set(EXPECTED_FIXED))}"
    )


def test_committed_labels_are_still_the_pre_fix_set(frame: pd.DataFrame) -> None:
    """The anchor: until a relabel lands as an ADDED artifact, labels.parquet
    keeps the heuristic set the audit measured. Overwriting it silently would
    erase the measurement this round exists to publish."""
    by_qid = dict(zip(frame["qid"].astype(str), frame["label"].astype(str), strict=True))
    for qid, expected in PRE_FIX_LABELS.items():
        assert by_qid[qid] == expected, f"{qid}: committed label changed from {expected!r}"


def test_audit_json_lower_bound_is_pinned() -> None:
    payload = json.loads((REPO / "data" / "label_error_audit.json").read_text(encoding="utf-8"))
    bound = payload["lower_bound_machine_label_error"]
    assert (bound["count"], bound["n"]) == (6, 120)
    assert bound["rate"] == pytest.approx(6 / 120)
