# Human labelling protocol for `data/human_validation_sample.csv`

Fill the `human_label` column only. One word per row: `correct` or `incorrect`
(case-insensitive, surrounding whitespace is fine). Anything else is rejected
by name and no agreement is reported until it is fixed. Leave the cell blank
when the row cannot be judged — a wrong gold list, an ambiguous question.
Blank rows are counted as unlabelled and excluded; they are not disagreements.

You are the ground truth this benchmark was missing. Every label in
`results.json` is machine-assigned, and the κ of 0.849 quoted there is
judge-versus-judge: it measures whether two judges share a blind spot, not
whether either is right. Your column is what bounds the machine error rate,
which bounds what any AUROC against those labels can mean
(`unc-bench human-agreement` prints the oracle ceiling once you save).

## The rules

Read `question`, `model_answer` and `gold_answers`. Judge the answer against
the gold list **as given**. If you think a gold alias is wrong, leave the row
blank rather than labelling against your own idea of the truth — the question
being measured is whether the judges applied *this* gold list the way a human
would.

1. **`correct`** — the answer states one of the gold answers. Extra words that
   do not change the claim are fine ("Bram Stoker" for gold "Stoker"; "It is
   Toronto" for gold "Toronto"). Transliterations and spelling variants count
   ("Chișinău" vs "Chisinau", "Rīga" vs "Riga"). A partial name counts when it
   unambiguously names the entity ("Clinton" for gold "Bill Clinton" counts;
   "Paris, Texas" for gold "Paris, France" does not).
2. **`incorrect`** — the answer names something else, hedges without
   committing, is empty, answers a different question, or echoes the question
   without answering it (see the echo rule below).
3. **Blank** — the gold list is wrong or incomplete in a way that makes the
   verdict a guess, or the question itself is ambiguous. Do not invent a third
   label.

## The inverse-relation echo rule

Several rows ask "What is X the capital of?" and the model answers "X"
("Rome" → "Rome", "Seattle" → "Seattle"). An echo is **not** an answer:

- Echo of a bare subject ("Rome" for "What is Rome the capital of?") is
  `incorrect`, even when the gold list happens to contain "Rome" among
  historical entities. Rome-the-city is not what Rome is the capital of.
- The one exception: if the gold list's correct reading genuinely is the
  subject itself (you judge the alias list, not the echo), mark `correct` —
  but expect this almost never to apply.

This is the row class that motivated the `capital of` removal for run #3
(`data/echo_contamination_report.json`): 14 of 20 echo rows in this file got
opposite labels for identical model behaviour, decided by alias-list
happenstance. Your verdicts here are the highest-value labels in the file.

## Worked examples from this file

- "What is Stockholm the capital of?" / answer "Stockholm" / gold contains
  "Stockholm County" and "Sweden": `incorrect` (echo; neither alias is
  Stockholm-the-city as an answer).
- "What is the capital of Ghana?" / answer "Ghana": `incorrect` (echo of the
  question's subject country, not its capital).
- "What is the capital of Chile?" / answer "Chilean capital is Santiago.":
  `correct` (extra words, same claim as gold "Santiago").

## When you are done

```bash
uv run unc-bench human-agreement --csv data/human_validation_sample.csv
```

It prints, for each of `heuristic_verdict`, `judge_primary_verdict`,
`judge_secondary_verdict` and `machine_label`: agreement rate, Cohen's κ with
a bootstrap interval, pooled minority counts — plus the label-noise ceiling on
observable AUROC. Labelling twenty rows and running the command reports
agreement over those twenty; partial passes are fine.
