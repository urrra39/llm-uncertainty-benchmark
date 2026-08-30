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
| `judge_secondary_verdict` | `claude-haiku-4-5`'s verdict; **empty for every row** — see below |
| `machine_label` | the label that entered the analysis, the one `results.json` counts |
| `machine_label_source` | `exact_match` or `judge` |
| `human_label` | **empty. For a human to fill in.** |

### Why `judge_secondary_verdict` is empty

`results.json` records that 66 rows went to both judges and that they agreed on
65 of them (`labels.kappa.observed_agreement` 0.9848 over `n` 66). It does not
record *which* row the single disagreement fell on, and the per-row judge
replies were not committed. So the secondary column cannot be filled from
committed artifacts. It is left empty and kept in the schema because that is
where those verdicts belong if the labelling stage is ever re-run; the
alternative — deriving it from the agreement rate — would mean inventing 65
verdicts to place one disagreement.

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
