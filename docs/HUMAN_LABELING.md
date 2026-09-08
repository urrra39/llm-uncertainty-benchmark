# Human labelling runbook: `unc-bench label-human --run run2b`

The command walks rows in a terminal. One word per row, nothing pre-filled,
resumable, with per-row timing. No model is called.

The published (primary) label set for run #2b is the **fixed** one — the
labels produced by the corrected rule in `data/run2b/labels_fixed.parquet`.
That is the label set a publishable ranking rests on, so the files a human
labels for run #2b are the `*_fixed.csv` targets below. The pre-fix files
(`data/human_validation_sample_run2b.csv`, `data/fuzzy_decided_rows.csv`)
are the Round-11 evidence snapshots; labelling them measures the old rule's
5.0% floor, but they are not the gate targets for the published set.

## Exact command

```bash
uv run unc-bench label-human --run run2b
```

That targets `data/fuzzy_decided_rows_fixed.csv` (73 rows) — the rows the
fuzzy rule decided under the fixed no-judge rule, which decide most of run
#2b's labels under a rule no human has checked. `--run run2b --target sample`
walks the 100-row fixed validation sample instead; any other value must be a
direct `.csv` path. Quitting (`q`) saves and exits 0 — rerunning resumes past
labelled rows without prompting.

## What you will see per row

The question, the gold alias list, and the model answer. NOT the machine
verdict: it appears only after you commit yours, so you cannot anchor on it.
(Your agreement with it is logged as feedback, not as a grade.)

## Keys

- `c` correct, `i` incorrect — written to `human_label` immediately.
- `a` ambiguous — stores a blank cell and logs the qid separately. Use it
  exactly when the gold list is wrong/incomplete or the question is
  ambiguous. In analysis, ambiguous rows are DROPPED (excluded from every
  AUROC and every agreement denominator), never coerced to either verdict:
  a forced label on an unjudgeable row is label noise by construction.
- `s` skip — leaves the row for later (it will be asked again on rerun).
- `q` quit — saves everything first.

## Adjudication rules

Judge the answer against the gold list **as given**. If a gold alias is wrong,
leave the row blank (or mark ambiguous) rather than labelling against your own
idea of the truth — the question being measured is whether the machine applied
*this* gold list the way a human would.

- **correct** — the answer states one of the gold answers. Extra words that do
  not change the claim are fine ("Bram Stoker" for gold "Stoker"; "It is
  Toronto" for gold "Toronto"). Transliterations and spelling variants count.
  A partial name counts when it unambiguously names the entity ("Clinton" for
  "Bill Clinton" counts; "Paris, Texas" for "Paris, France" does not). A full
  gold alias stated verbatim with extra words that do not change the claim is
  the "verbose but correct" shape the fixed rule accepts.
- **incorrect** — names something else, hedges without committing, is empty,
  answers a different question, or echoes the question without answering it.
- **Echo rule** (the highest-value rows in this file): several rows ask "What
  is X the capital of?" and the model answers "X". An echo is **not** an
  answer — mark `incorrect`, even when the gold list happens to contain the
  subject among historical entities. Worked case in this file's sibling
  sample: "What is Stockholm the capital of?" / "Stockholm" → incorrect.
- **Ambiguous gold / unjudgeable row** — leave the cell blank (mark `a`). The
  one row in `data/fuzzy_decided_rows_fixed.csv` whose `fuzzy_verdict` is
  blank (`triviaqa-jp_1520`, gold includes "Unicorn Whale", model answered
  "whale") is exactly such a case: the fixed rule refused to decide it, and it
  is queued for a human. If you judge it, write `correct` or `incorrect`; if
  you cannot, leave it blank. It is never coerced.

## Duration

The minimum honest total is **59 rows**: the coverage gate needs 59 of the 73
fuzzy rows, the protocol gate needs 50 of the 100 sample rows, and 54 rows
sit in both files — so label the shared rows first (`unc-bench label-plan`
prints the exact order). Per-row timing is unmeasured (no human has done
this yet); at 20–40 seconds per row that is roughly 20–40 minutes, but treat
that band as a guess, not a measurement — the timing log beside the CSV
records the real median from row one. Partial passes are fine: unlabelled
rows are counted and skipped. Gate/file mapping:
[LABEL_GATE_MAP.md](LABEL_GATE_MAP.md) (generated, pinned by test).

## When you are done

```bash
uv run unc-bench human-agreement --csv data/human_validation_sample_run2b_fixed.csv
```

It prints, for each machine column present: agreement rate, Cohen's κ with a
bootstrap interval and minority counts, plus the label-noise ceiling on
observable AUROC. Then re-run `unc-bench analyze --config
configs/run2b_fixedlabels.yaml` so `label_quality` and the human gates reflect
the labels — only then is the ranking publishable.
