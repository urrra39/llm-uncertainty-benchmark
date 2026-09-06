# Human labelling runbook: `unc-bench label-human --run run2b`

The command walks rows in a terminal. One word per row, nothing pre-filled,
resumable, with per-row timing. No model is called.

## Exact command

```bash
uv run unc-bench label-human --run run2b
```

That targets `data/fuzzy_decided_rows.csv` (73 rows) — the rows where the
label risk actually lives: they decided 61% of run #2b's labels under a rule
no human has checked, while the 47 exact-match rows are a deterministic
string comparison that needs no human. `--run run2b --target sample` walks
the 100-row validation sample instead; any other value must be a direct
`.csv` path. Quitting (`q`) saves and exits 0 — rerunning resumes past
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

## Duration

73 fuzzy-decided rows at a measured median of ~20 seconds per row is roughly
25 minutes. The timing log beside the CSV (`.timing.json`) records the real
median; if your median differs, budget from it, not from this paragraph.
Partial passes are fine: unlabelled rows are counted and skipped. The full
100-row sample costs proportionally more and adds mostly exact-match rows of
no labelling value — do the fuzzy file first.

## Adjudication rules

Judge the answer against the gold list **as given**. If a gold alias is wrong,
leave the row blank (or mark ambiguous) rather than labelling against your own
idea of the truth — the question being measured is whether the machine applied
*this* gold list the way a human would.

- **correct** — the answer states one of the gold answers. Extra words that do
  not change the claim are fine ("Bram Stoker" for gold "Stoker"; "It is
  Toronto" for gold "Toronto"). Transliterations and spelling variants count.
  A partial name counts when it unambiguously names the entity ("Clinton" for
  "Bill Clinton" counts; "Paris, Texas" for "Paris, France" does not).
- **incorrect** — names something else, hedges without committing, is empty,
  answers a different question, or echoes the question without answering it.
- **Echo rule** (the highest-value rows in this file): several rows ask "What
  is X the capital of?" and the model answers "X". An echo is **not** an
  answer — mark `incorrect`, even when the gold list happens to contain the
  subject among historical entities. Worked case in this file's sibling
  sample: "What is Stockholm the capital of?" / "Stockholm" → incorrect.

## When you are done

```bash
uv run unc-bench human-agreement --csv data/human_validation_sample_run2b.csv
```

It prints, for each machine column: agreement rate, Cohen's κ with a bootstrap
interval and minority counts, plus the label-noise ceiling on observable
AUROC. Then re-run `unc-bench analyze --config configs/run2b_clean.yaml` so
`label_quality` and the human gates reflect the labels — only then is the
ranking publishable.
