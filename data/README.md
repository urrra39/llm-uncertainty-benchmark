# data/

## `human_validation_sample.csv`

A 100-row sample of run #2's labelled rows, laid out for a human to check the
machine labels against the gold answers. **The `human_label` column is empty on
purpose.** No human has labelled any row in this project, so filling the column
with anything other than a real human's judgement would be fabrication. It is
shipped empty and the tooling reports "no human labels present" rather than a
number.

The sample is 100 of the run's 120 rows (79 PopQA, 21 TriviaQA), balanced 50/50
on the machine label so a labeller sees an equal number of each verdict.

### Columns

| column | meaning |
|---|---|
| `qid` | row identifier, `<dataset>-<source id>` |
| `dataset` | `popqa` or `triviaqa` |
| `question` | the question put to the model under test |
| `gold_answers` | every accepted gold alias, ` \| `-separated |
| `model_answer` | what Qwen2.5-0.5B-Instruct answered under greedy decoding |
| `heuristic_verdict` | the no-judge label: normalized exact match, then two-way token-sequence containment |
| `judge_primary_verdict` | `gpt-5-mini`'s verdict; empty where exact match settled the row and no judge was called |
| `judge_secondary_verdict` | `claude-haiku-4-5`'s verdict; filled on the 53 judge-settled rows, empty on the 47 rows exact match settled — see below |
| `machine_label` | the label that entered the analysis, the one `results.json` counts |
| `machine_label_source` | `exact_match` or `judge` |
| `human_label` | **empty. For a human to fill in.** |

### How `judge_secondary_verdict` was recovered, and why 47 cells are still empty

`results.json` records that 66 rows went to both judges and that they agreed on
65 of them (`labels.kappa.observed_agreement` 0.9848 over `n` 66). It does not
record *which* row the single disagreement fell on, and run #2's per-row judge
replies were never committed.

The second judge is a pure function of (question, gold aliases, model answer)
under the frozen rubric in `unc_bench.labeling.JUDGE_PROMPT`, so a row can be
re-judged whenever those three inputs survive. They survive for exactly the rows
in this file: `data/run2/` was never committed, so the `model_answer` column here
is the only committed record of any run #2 answer.

`scripts/recover_secondary_verdicts.py` re-ran `claude-haiku-4-5` at temperature
0 over the 53 rows whose `machine_label_source` is `judge`, and those cells are
now filled. It recovered 53 of 53 with no parse failures. The 47 `exact_match`
rows are blank because no judge was ever asked about them — that blank is
correct, not missing.

Two limits worth stating plainly:

- **The recovered κ is over 53 rows, not 66.** 13 of run #2's judged rows fall
  outside this 100-row sample, and their model answers exist nowhere in the
  repository, so they cannot be re-judged without re-running generation. The
  script asserts rather than assumes that the 53 are a subset of the 66: with
  `judges.cross_validation_n` at 120 against 66 judged rows, the second judge
  saw all of them.
- **κ over the recovered subset is 1.0000, against the stored 0.8493 on n=66.**
  These are different estimators over different row sets and the gap is
  arithmetic, not judge instability: run #2 recorded exactly one disagreement,
  and it lies among the 13 rows that cannot be reproduced. Every one of the 53
  recovered verdicts matches the primary verdict recorded in run #2, so no
  instability was observed on any row that could be re-asked. `results.json`
  keeps 0.8493 as the published value; the recovered figure is reported beside
  it in `data/judge_verdicts_recovered.json` and is not a replacement for it.

The underlying cause is fixed for future runs. The `label` stage now writes both
verdict columns into its labels checkpoint and a standalone `judge_verdicts.json`
naming every row either judge saw, so no later run can quote a κ it cannot
recompute.

`judge_primary_verdict` is recoverable because on a judged row the primary
judge's verdict *is* the machine label: nothing else could have set it, given
`labels.heuristic_fallback_rows` is 0 and `judge_parse_failures` is 0.

### How `gold_answers` was reconstructed

The column shipped corrupted. A Python list had been iterated character-wise
and joined with ` | `, then truncated to 17 characters, so every row read
`[ | " | S | t | o` — the first five characters of the repr. The information
was not in the file.

It was recovered by looking each `qid` up in the two source corpora (PopQA
`test.tsv`, TriviaQA `rc.nocontext` validation) through the project's own
dataset builders, which is a read of static published ground truth and not a
regeneration of any model output. Two checks guard the recovery, both of which
must pass:

- the recovered question text matches the CSV's question text exactly, for all
  100 rows;
- every one of the 47 `exact_match` rows still reproduces as `correct` under
  the project's own matcher against the recovered aliases.

### Labelling convention

Read `question`, `model_answer` and `gold_answers`. Write `correct` or
`incorrect` in `human_label`. Nothing else is a valid value.

- **`correct`** — the answer states one of the gold answers. Extra words that do
  not change the claim are fine ("Bram Stoker" for gold "Stoker"; "It is
  Toronto" for gold "Toronto").
- **`incorrect`** — the answer names something else, hedges without committing,
  is empty, or answers a different question than the one asked.
- **Leave the cell blank** if the row cannot be judged — a gold list that is
  wrong, an ambiguous question. Blank rows are counted as unlabelled and
  excluded; they are not treated as disagreements. Do not invent a third label.

Judge both against the gold list as given. If you think a gold alias is wrong,
leave the row blank rather than labelling against your own idea of the truth —
the question being measured is whether the judges applied *this* gold list the
way a human would.

One thing worth knowing before you start: the first row of the file is a case
where the heuristic and the judge disagree. Gold for "What is Stockholm the
capital of?" includes both "Stockholm County" and "Sweden"; the model answered
"Stockholm". Containment scores that correct, the judge called it incorrect.
That row is the kind of thing this file exists to settle.

### Feeding labels back

With the column filled in:

```bash
unc-bench human-agreement --csv data/human_validation_sample.csv
```

It prints, for each of `heuristic_verdict`, `judge_primary_verdict`,
`judge_secondary_verdict` and `machine_label`, the agreement rate against
`human_label` and Cohen's κ, over the rows where both sides have a verdict. It
reads the CSV and prints; it writes nothing, calls no model, and does not touch
`results.json`.

Labelling one row takes well under a second — read three short strings, type one
word — so a full pass over the 100 rows is a couple of minutes. Partial passes
are fine: unlabelled rows are counted and skipped, so labelling twenty rows and
running the command reports agreement over those twenty.

Rows with an unrecognised `human_label` are rejected by name and no agreement
is reported until they are fixed. Rows with a blank `human_label` are skipped
and counted.

Nothing downstream consumes the output automatically. Changing the run's labels
would mean re-running the analysis stage, which would move every published
number, so that is a deliberate decision and not a side effect of filling in a
CSV. What the utility gives you is the number needed to say whether the machine
labels are trustworthy — which is the claim `results.json` currently cannot
support in either direction.

## `run2_pilot/dataset.parquet`

The 40-row pilot question set that chose run #2's dataset mix. Questions and
gold answers only, no generations. Kept because `docs/DECISIONS.md` cites its
measured base rates.

## What is not here

Generations, signal columns and the judge cache are not committed —
`.gitignore` excludes `data/*.parquet`, `data/artifacts/`, `data/cache/` and
`data/raw/`. One consequence matters for reading the results: the per-row signal
values no longer exist in the repository, so anything needing them cannot be
recomputed from a clone. `results.json` stores the derived quantities — the
21×21 Spearman matrix, per-signal bootstrap intervals, per-dataset point
estimates — and that is what `unc-bench audit` reads. See `docs/DECISIONS.md`
for what this rules out.
