"""Re-run the second judge over run #2's judged rows and persist per-row verdicts.

Why this script exists. Run #2 stored Cohen's kappa as an aggregate — 66 rows,
65 agreements, kappa 0.849 — and did not persist the per-row second-judge
replies. `data/human_validation_sample.csv` therefore shipped with
`judge_secondary_verdict` empty on every row: the agreement rate was known, the
row the single disagreement fell on was not. The underlying cause is fixed in
`unc_bench.stages.label`, which now writes `judge_verdicts.json` and carries both
verdict columns in the labels checkpoint for every future run. This script
recovers what it can for the run that predates that fix.

What it can and cannot recover, stated exactly.

The second judge is a pure function of (question, gold aliases, model answer)
plus the frozen rubric in `unc_bench.labeling.JUDGE_PROMPT`. Two of those three
inputs are static published ground truth and are re-derivable through the
project's own dataset builders; the third, the model answer, is a stored output
of run #2's generation pass. So a row can be re-judged if and only if its model
answer survives somewhere in the repository.

`data/run2/` was never committed — `.gitignore` excludes `data/*.parquet` and
`data/artifacts/`, and `git log --all --diff-filter=A` over `data/` returns three
files, none of them a generation. The only committed record of any run #2 model
answer is the `model_answer` column of the 100-row validation CSV. 53 of those
100 rows are `machine_label_source == judge`, and those 53 are exactly the rows
this script can re-judge. The other 13 of the run's 66 judged rows are outside
the CSV sample and their model answers do not exist in the repository, so they
are not recoverable by any means short of re-running generation, which the run's
terms forbid.

Every judged row went to the second judge in run #2: `judges.cross_validation_n`
is 120 and only 66 rows were judged, so `cross_validation_sample` selected all of
them. That is what makes the 53 a genuine subset of the 66 rather than a
differently-drawn sample, and it is asserted below rather than assumed.

Consequently the kappa this script computes is over 53 rows, not 66. It is
reported as such and does not replace the stored value; both are published. A
53-row kappa is a different estimator from a 66-row kappa even when the judges
are perfectly stable, so agreement between the two numbers is evidence and
disagreement is not proof of instability.

Costs 53 calls to claude-haiku-4-5 at temperature 0. Verdicts are written
through the same `JudgeCache` the label stage uses, so a rerun is free.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from unc_bench.config import Config
from unc_bench.labeling import (
    build_judge,
    cohens_kappa,
    cross_validation_sample,
)
from unc_bench.stages.label import JudgeCache, cached_judge
from unc_bench.types import Question

#: Column the recovered verdicts are written into.
SECONDARY_COLUMN = "judge_secondary_verdict"

#: Only these rows saw a judge at all. An `exact_match` row was settled for free
#: and no judge was ever asked about it, so an empty secondary verdict there is
#: correct rather than missing.
JUDGED_SOURCE = "judge"

#: Alias separator used by the validation CSV's `gold_answers` column.
ALIAS_SEPARATOR = " | "


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def question_from_row(row: dict[str, str]) -> Question:
    """Rebuild the judge's input from the CSV row.

    The gold list is taken from the CSV rather than re-derived from the corpora,
    because the CSV column is what a reader of this repository can inspect, and
    `data/README.md` already documents the two checks that established it
    matches the source corpora.
    """
    aliases = tuple(a.strip() for a in row["gold_answers"].split(ALIAS_SEPARATOR) if a.strip())
    return Question(
        qid=row["qid"],
        dataset=row["dataset"],
        question=row["question"],
        gold_answers=aliases,
    )


def recover(cfg: Config, csv_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Re-judge every recoverable row and return a report.

    Writes the recovered verdicts back into the CSV and into
    `judge_verdicts_recovered.json` beside it, unless `dry_run`.
    """
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    if SECONDARY_COLUMN not in rows[0]:
        raise KeyError(f"{csv_path} has no {SECONDARY_COLUMN!r} column")

    judged = [r for r in rows if r["machine_label_source"] == JUDGED_SOURCE]
    if not judged:
        raise ValueError(f"{csv_path} contains no {JUDGED_SOURCE!r} rows to re-judge")

    # Assert the premise that makes these rows a subset of the run's kappa set:
    # with cross_validation_n >= the number of judged rows, every judged row was
    # sent to the second judge, so no row here was excluded by sampling.
    if cfg.judges.cross_validation_n < len(judged):
        raise AssertionError(
            f"cross_validation_n {cfg.judges.cross_validation_n} is below the "
            f"{len(judged)} judged rows, so the second judge saw only a sample "
            "and these rows cannot be assumed to be part of it"
        )
    selected = set(cross_validation_sample(sorted(r["qid"] for r in judged), cfg.judges))
    unselected = sorted({r["qid"] for r in judged} - selected)
    if unselected:
        raise AssertionError(f"rows not in the second-judge sample: {unselected[:5]}")

    report: dict[str, Any] = {
        "csv": str(csv_path),
        "secondary_model": cfg.judges.secondary.name,
        "secondary_seed": cfg.judges.secondary_seed,
        "n_csv_rows": len(rows),
        "n_judged_rows": len(judged),
        "dry_run": dry_run,
    }
    if dry_run:
        report["verdicts"] = {}
        return report

    cache = JudgeCache(cfg.paths.artifacts_dir / "judge_cache.json")
    judge = build_judge(cfg, secondary=True)

    verdicts: dict[str, str] = {}
    raws: dict[str, str] = {}
    parse_failures: list[str] = []
    for index, row in enumerate(judged, start=1):
        question = question_from_row(row)
        answer = row["model_answer"]
        outcome = cached_judge(
            judge, cache, question, answer, seed=cfg.judges.secondary_seed
        )
        if outcome.value is None:
            parse_failures.append(question.qid)
        else:
            verdicts[question.qid] = outcome.value
        raws[question.qid] = outcome.raw[:200]
        if index % 10 == 0:
            cache.flush()
            print(f"[recover] {index}/{len(judged)}", flush=True)
    cache.flush()

    # Kappa over the rows where both judges have a verdict. The primary verdict
    # is read from the CSV, where it was recovered from the machine label; see
    # data/README.md for why that identification is sound.
    paired = [r for r in judged if r["qid"] in verdicts]
    primary_side = [r["judge_primary_verdict"] for r in paired]
    secondary_side = [verdicts[r["qid"]] for r in paired]
    result = cohens_kappa(
        primary_side, secondary_side, threshold=cfg.judges.kappa_trust_threshold
    )
    disagreements = [
        r["qid"] for r in paired if r["judge_primary_verdict"] != verdicts[r["qid"]]
    ]

    report.update(
        {
            "n_recovered": len(verdicts),
            "n_parse_failures": len(parse_failures),
            "parse_failures": parse_failures,
            "kappa": result.kappa,
            "kappa_n": result.n,
            "observed_agreement": result.observed_agreement,
            "expected_agreement": result.expected_agreement,
            "trustworthy": result.trustworthy,
            "n_agreements": result.n - len(disagreements),
            "disagreement_qids": disagreements,
            "verdicts": verdicts,
            "raw_replies": raws,
        }
    )

    for row in rows:
        if row["qid"] in verdicts:
            row[SECONDARY_COLUMN] = verdicts[row["qid"]]

    fieldnames = list(rows[0])
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)

    out = csv_path.parent / "judge_verdicts_recovered.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["written_to"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/run2.yaml"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be re-judged without calling the judge",
    )
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    csv_path = args.csv if args.csv is not None else cfg.paths.human_validation_csv
    report = recover(cfg, Path(csv_path), dry_run=bool(args.dry_run))

    print(f"rows in CSV      : {report['n_csv_rows']}")
    print(f"judged rows      : {report['n_judged_rows']}")
    if report["dry_run"]:
        print("dry run: no judge calls made")
        return 0
    print(f"recovered        : {report['n_recovered']}")
    print(f"parse failures   : {report['n_parse_failures']}")
    print(f"kappa            : {report['kappa']:.4f} on n={report['kappa_n']}")
    print(f"observed agree   : {report['observed_agreement']:.4f}")
    print(f"disagreements    : {report['disagreement_qids']}")
    print(f"written to       : {report['written_to']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
