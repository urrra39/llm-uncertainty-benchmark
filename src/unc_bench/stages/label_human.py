"""Terminal labelling loop for a human (Part P2.1).

`unc-bench label-human --run run2b` walks the run's validation CSV row by row
and asks a human for each verdict. Rules that protect the measurement:

- Never pre-fills: rows already labelled are skipped without prompting.
- Never shows the machine label until AFTER the human commits a verdict, so
  the human cannot anchor on it. Agreement is shown after commit, as feedback.
- Writes the CSV atomically after every verdict, so quitting loses nothing
  and rerunning resumes where it stopped.
- Logs per-row timing plus agreement to a sibling `.timing.json`, which is
  what the runbook's duration estimate is computed from.
- Accepts correct / incorrect / ambiguous / skip / quit. `ambiguous` stores a
  blank cell (the scorer only accepts correct/incorrect) and records the qid
  in the timing log's explicitly-ambiguous list.

No model is called, nothing is imported beyond the standard library plus
pandas for the CSV round-trip.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

#: Map from --run names to validation CSVs. Withdrawn runs are deliberately
#: absent: labelling withdrawn rows earns no gate credit, so they are not a
#: target. A direct .csv path still works as an escape hatch for anything.
#:
#: Run #2b's published (primary) label set is the FIXED one, so the gate files
#: a human labels for run #2b are the *_fixed.csv targets, not the pre-fix
#: snapshots (which remain committed as the object of the labeler measurement).
RUN_CSVS = {
    "run2b": Path("data/human_validation_sample_run2b_fixed.csv"),
}

#: Default labelling target: the rows where the label risk actually lives.
#: The 73 fuzzy-decided rows decide most of run #2b's labels under a rule no
#: human has checked, so they outrank the exact-match rows (a deterministic
#: string comparison needs no human) in labelling value per minute. The fixed
#: set is the published one; data/fuzzy_decided_rows.csv is the pre-fix
#: snapshot kept as evidence.
DEFAULT_TARGET = Path("data/fuzzy_decided_rows_fixed.csv")

VALID_VERDICTS = ("correct", "incorrect")


def label_plan(sample_csv: str | Path, fuzzy_csv: str | Path) -> dict[str, object]:
    """The minimum labelling order that greens every label gate (Part B4).

    Protocol gate needs 50 of the sample's 100 rows; coverage gate needs 59 of
    the fuzzy file's 73 rows. Rows in both files count toward both gates, so
    the plan labels shared rows first, then fuzzy-only rows until the coverage
    floor, then sample-only rows if the protocol floor is not yet met.
    Unlabelled counts assume a fresh start; already-labelled rows are
    subtracted, never re-planned. Pure function of the two committed files.
    """
    import pandas as pd

    sample = pd.read_csv(sample_csv, dtype=str, keep_default_na=False)
    fuzzy = pd.read_csv(fuzzy_csv, dtype=str, keep_default_na=False)
    sample_qids = [str(q) for q in sample["qid"]]
    fuzzy_qids = [str(q) for q in fuzzy["qid"]]
    sample_done = {
        str(q) for q, h in zip(sample["qid"], sample["human_label"], strict=True) if str(h).strip()
    }
    fuzzy_done = {
        str(q) for q, h in zip(fuzzy["qid"], fuzzy["human_label"], strict=True) if str(h).strip()
    }
    shared = [q for q in fuzzy_qids if q in set(sample_qids)]
    shared_todo = [q for q in shared if q not in fuzzy_done and q not in sample_done]
    fuzzy_only = [q for q in fuzzy_qids if q not in set(sample_qids) and q not in fuzzy_done]
    sample_only = [q for q in sample_qids if q not in set(fuzzy_qids) and q not in sample_done]
    order = shared_todo + fuzzy_only + sample_only
    # Gates need 59 fuzzy rows and 50 sample rows; shared rows serve both.
    # Take shared first, then fuzzy-only to 59 fuzzy, then sample-only to 50.
    plan: list[str] = []
    fuzzy_count = len(fuzzy_done)
    sample_count = len(sample_done)
    for qid in order:
        if fuzzy_count >= 59 and sample_count >= 50:
            break
        plan.append(qid)
        if qid in set(fuzzy_qids) - fuzzy_done:
            fuzzy_count += 1
        if qid in set(sample_qids) - sample_done:
            sample_count += 1
    return {
        "rows_to_label": len(plan),
        "file_order": plan,
        "fuzzy_coverage_after": [fuzzy_count, 73],
        "sample_coverage_after": [sample_count, 100],
        "wall_clock": (
            f"{len(plan)} rows x per-row rate (unmeasured; roughly "
            f"{len(plan) * 20 // 60}-{len(plan) * 40 // 60} min at 20-40 s/row)"
        ),
    }


def resolve_csv(run: str, target: str = "fuzzy_decided") -> Path:
    """Resolve the labelling target. `--run` picks the run, `--target` picks
    the population: `fuzzy_decided` (default: the rows the fuzzy rule
    decided) or `sample` (the run's validation CSV). Anything else is a
    direct .csv path. Raises rather than guessing."""
    if target == "fuzzy_decided" and run == "run2b":
        return DEFAULT_TARGET
    if target not in ("sample", "fuzzy_decided"):
        candidate = Path(target)
        if candidate.suffix == ".csv":
            return candidate
        raise ValueError(f"unknown target {target!r}: use fuzzy_decided, sample or a .csv path")
    if run in RUN_CSVS:
        return RUN_CSVS[run]
    candidate = Path(run)
    if candidate.suffix == ".csv":
        return candidate
    raise ValueError(f"unknown run {run!r}: use one of {sorted(RUN_CSVS)} or a direct .csv path")


def run_loop(
    csv_path: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run the labelling loop. Returns a summary; used by the CLI and by tests."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"no validation CSV at {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"qid", "question", "gold_answers", "model_answer", "human_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    timing_path = path.with_suffix(".timing.json")
    timing: dict[str, Any] = {"rows": {}, "explicitly_ambiguous": []}
    if timing_path.exists():
        try:
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            timing = {"rows": {}, "explicitly_ambiguous": []}

    machine_col = "machine_label" if "machine_label" in frame.columns else None
    session_labelled = 0
    skipped = 0
    for position in range(len(frame)):
        row = frame.iloc[position]
        if str(row["human_label"]).strip():
            continue
        print(f"\n--- {row['qid']} ({position + 1}/{len(frame)}) ---")
        print(f"Q: {row['question']}")
        print(f"gold: {row['gold_answers']}")
        print(f"model: {row['model_answer']}")
        started = clock()
        verdict: str | None = None
        while True:
            reply = input_fn("[c]orrect/[i]ncorrect/[a]mbiguous/[s]kip/[q]uit: ").strip().lower()
            if reply in ("c", "correct"):
                verdict = "correct"
                break
            if reply in ("i", "incorrect"):
                verdict = "incorrect"
                break
            if reply in ("a", "ambiguous"):
                verdict = ""
                break
            if reply in ("s", "skip"):
                verdict = None
                break
            if reply in ("q", "quit"):
                _save(frame, timing, timing_path, path)
                return _summary(frame, timing, session_labelled, quit_early=True)
            print("unrecognised; type c, i, a, s or q")
        elapsed = clock() - started
        if verdict is None:
            skipped += 1
            continue
        frame.at[frame.index[position], "human_label"] = verdict
        entry: dict[str, Any] = {"seconds": round(elapsed, 1)}
        if verdict == "":
            timing["explicitly_ambiguous"].append(str(row["qid"]))
            print("recorded as ambiguous (blank cell, logged).")
        else:
            session_labelled += 1
            if machine_col is not None:
                machine = str(row[machine_col]).strip().lower()
                entry["agreed_with_machine"] = machine == verdict
                print(f"recorded {verdict}. machine said {machine or '?'} — logged.")
            else:
                print(f"recorded {verdict}.")
        timing["rows"][str(row["qid"])] = entry
        _save(frame, timing, timing_path, path)
    return _summary(frame, timing, session_labelled, quit_early=False)


def _save(frame: pd.DataFrame, timing: dict[str, Any], timing_path: Path, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True), encoding="utf-8")


def _summary(
    frame: pd.DataFrame, timing: dict[str, Any], session_labelled: int, *, quit_early: bool
) -> dict[str, Any]:
    filled = int((frame["human_label"].astype(str).str.strip() != "").sum())
    times = [r["seconds"] for r in timing["rows"].values() if "seconds" in r]
    summary = {
        "labelled_this_session": session_labelled,
        "labelled_total": filled,
        "n_rows": int(len(frame)),
        "skipped_ambiguous_logged": len(timing["explicitly_ambiguous"]),
        "median_seconds_per_row": (float(sorted(times)[len(times) // 2]) if times else None),
        "quit_early": quit_early,
    }
    print(
        f"\nlabelled {session_labelled} this session, {filled}/{len(frame)} total; "
        f"timing log beside the CSV"
    )
    return summary
