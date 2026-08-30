# Which cheap uncertainty signal best predicts an LLM's factual errors?

Twenty-one uncertainty signals from three families — token logprobs (1× cost),
self-consistency over 5 samples (6× cost), and self-verification / P(True)
(2× cost) — ranked by how well each predicts that Qwen2.5-0.5B-Instruct got a
short factual question wrong. n=120 questions from PopQA and TriviaQA,
correctness labeled by exact match plus two LLM judges.

**ALL THREE VALIDITY GATES PASS.** Random baseline `t_random` AUROC
**0.508 [0.404, 0.611]** (CI contains 0.50, as it must). **63 incorrect / 57
correct** (both classes ≥ 30). Abstention rate **0/120 = 0.000** (below the 0.10
ceiling). The ranking below is publishable. Run #1's ranking was not; see the
last section.

## Findings

**1. No signal separated from the others at n=120.** The top six span AUROC
0.704 to 0.691 and every one of their confidence intervals overlaps every other
one's. Of 20 paired comparisons against the leader, 4 are significant after
Holm–Bonferroni and all 4 are signals that are *worse*. Nothing is significantly
better than anything else. The honest headline is not "self-consistency wins",
it is "this benchmark could not tell these signals apart".

**2. A trivial question-length baseline nearly matched 6×-cost
self-consistency, and the gap is not significant.** `t_question_length` — the
number of words in the question, computed from text already in hand, costing
**1×** — scores **0.684**. The leader `b_distinct_count` costs **6×** (five extra
generations plus an NLI clustering pass) and scores **0.704**. The paired
bootstrap puts that 0.020 gap at **p_holm = 1.0**. Six times the compute bought
an increase that this study cannot distinguish from zero, against a baseline
that is a property of the *input* and knows nothing about the model's
uncertainty.

**3. Self-consistency saturates at N=3; the last two samples are wasted.**
AUROC by sample count is 0.628 / 0.678 / **0.705** / 0.704 at N = 1/2/3/5. The
curve is flat from N=3 onward — N=5 is very slightly *lower* — so two fifths of
family B's cost buys nothing measurable. Anyone who does deploy
self-consistency should run three samples, not five.

**4. The per-dataset split changes the ranking, so the pooled table is
flattered by the dataset mix.** PopQA (n=90) is 40% incorrect; TriviaQA (n=30)
is 90% incorrect. A signal that merely separates the two datasets gains pooled
AUROC without predicting anything. The leader falls from **0.704 pooled to
0.603 on PopQA alone**, and the pooled top-six ordering does not survive the
split: within PopQA the best signal is `c_p_true_plain` (0.626), within TriviaQA
it is `c_verbal_confidence` (0.765) — the same signal that scores 0.484 pooled,
below chance. The per-dataset table is the more honest read of this benchmark.

The practical conclusion, argued in full under
["What I would ship"](#what-i-would-ship): this benchmark does not establish any
of the 21 signals above chance on its larger, more balanced subset, so it does
not support shipping any of them at this scale.

## Per-dataset AUROC

The primary table. Run #2, n=120, `configs/run2.yaml`, positive class
**incorrect answer**. Two datasets with very different base rates, scored
separately so no signal can earn credit for telling them apart.

| signal | family | cost | PopQA AUROC (n=90, 40% incorrect) | TriviaQA AUROC (n=30, 90% incorrect) | pooled AUROC |
|---|---|---|---|---|---|
| `b_distinct_count` | B | 6.0× | 0.603 | 0.741 | 0.704 |
| `b_distinct_fraction` † | B | 6.0× | 0.603 | 0.741 | 0.704 |
| `b_disagreement_rate` | B | 6.0× | 0.601 | 0.685 | 0.701 |
| `b_semantic_entropy` | B | 6.0× | 0.579 | 0.704 | 0.693 |
| `b_semantic_entropy_normalized` † | B | 6.0× | 0.579 | 0.704 | 0.693 |
| `b_mean_pairwise_f1` | B | 6.0× | 0.595 | 0.691 | 0.691 |
| `t_question_length` | T | 1.0× | 0.505 | 0.580 | 0.684 |
| `a_mean_logprob` | A | 1.0× | 0.514 | 0.753 | 0.664 |
| `c_p_true_plain` | C | 2.0× | **0.626** | 0.660 | 0.659 |
| `t_random` | T | 1.0× | 0.511 | 0.691 | 0.508 |
| `c_verbal_confidence` | C | 2.0× | 0.395 | **0.765** | 0.484 |

Rows marked † are rank-equivalent duplicates of the row above (see
[the pooled table](#auroc-and-auprc-all-21-signals)); the 11 rows are 9 distinct
signals.

Three things in that table are worth stating plainly:

- **The leader loses most of its edge on PopQA.** 0.704 pooled, 0.603 on the
  90-row subset. Family B's pooled strength is substantially the dataset mix.
- **`t_question_length` is at chance within PopQA** (0.505) and barely above it
  within TriviaQA (0.580), yet scores 0.684 pooled. That is the dataset-mix
  effect in its purest form: question length separates PopQA from TriviaQA, and
  the datasets differ in base rate.
- **`t_random` scores 0.691 within TriviaQA.** A random number cannot predict
  anything. With 3 correct rows in a 30-row subset, that is what estimator noise
  looks like — which is exactly the failure that invalidated run #1, and the
  reason the TriviaQA column should not be read as a ranking.

Per-dataset AUPRC is in `results.json` under
`views.primary.per_dataset.datasets.<name>.signals`.

## Supporting results (pooled)

### AUROC and AUPRC, all 21 signals

Every number below was computed in this run on the identical 120-row frozen
analysis set (`qid_digest = ffff86216137caed`). Positive class is **incorrect
answer**. Higher is better. AUPRC baseline (the prevalence of the positive
class) is **0.525** — an AUPRC at or below that number is no better than
guessing. Read this table alongside the per-dataset table above, not instead of
it.

**The 21 rows are 18 distinct signals.** AUROC is invariant under any strictly
monotone transform of the score, so a signal that is a monotone
reparameterization of another produces an identical ranking and an identical
AUROC — it is the same signal wearing a different unit. Three pairs are
rank-equivalent, each verified by measuring the Spearman rank correlation over
the 120 frozen rows and finding it exactly **+1.000** (`views.primary.correlation`
in `results.json`; reproduce with `unc-bench audit`):

| pair | relationship | measured Spearman |
|---|---|---|
| `a_mean_logprob` / `a_perplexity` | `exp(-x)`, strictly monotone | +1.000 |
| `b_distinct_count` / `b_distinct_fraction` | divide by the answer-set size, constant at 6 | +1.000 |
| `b_semantic_entropy` / `b_semantic_entropy_normalized` | divide by `ln(6)`, constant *in this run only* | +1.000 |

The third pair is not a monotone transform in general — dividing by
`ln(answer-set size)` is only monotone when that size is fixed. Here it is:
`n_samples = 5` and the answer set is `[greedy, *samples]`, so every row divides
by `ln(6)`. This is a property of the configuration, not of the signal, and it
is the reason the tie is expected rather than a bug. See
[docs/DECISIONS.md](docs/DECISIONS.md) for the code-level diagnosis.

Consequence for the significance table below: Holm–Bonferroni was applied across
**20 comparisons, of which 3 are rank-equivalent duplicates of others in the
same family**. A duplicate carries no information the family did not already
have, so the effective family is 17 and the correction is **conservative** — it
adjusts p-values upward more than the evidence requires, which makes
non-significance easier to obtain than it should be. The bias runs against the
signals, not in their favour.

The deduplicated correction is derivable from the stored bootstrap p-values,
since only the family size changes, so both are reported. Recomputing Holm over
the 17 non-duplicate comparisons **changes no verdict**: the same 4 comparisons
are significant either way, and the largest movement is `a_min_logprob` from
1.000 to 0.974, still far from 0.05. **The original 20-comparison correction
therefore remains primary** and no published p-value changes. Full side-by-side
table: `unc-bench audit`.

Rows marked † are rank-equivalent duplicates of the row immediately above them.

| signal | family | AUROC [95% CI] | AUPRC [95% CI] | extra calls/q | cost |
|---|---|---|---|---|---|
| `b_distinct_count` | B | 0.704 [0.616, 0.786] | 0.724 [0.651, 0.796] | 5 | 6.0× |
| `b_distinct_fraction` † | B | 0.704 [0.616, 0.786] | 0.724 [0.651, 0.796] | 5 | 6.0× |
| `b_disagreement_rate` | B | 0.701 [0.612, 0.783] | 0.716 [0.636, 0.795] | 5 | 6.0× |
| `b_semantic_entropy` | B | 0.693 [0.604, 0.774] | 0.712 [0.635, 0.789] | 5 | 6.0× |
| `b_semantic_entropy_normalized` † | B | 0.693 [0.604, 0.774] | 0.712 [0.635, 0.789] | 5 | 6.0× |
| `b_mean_pairwise_f1` | B | 0.691 [0.600, 0.775] | 0.714 [0.635, 0.794] | 5 | 6.0× |
| `t_question_length` | T | 0.684 [0.599, 0.763] | 0.697 [0.618, 0.788] | 0 | 1.0× |
| `a_mean_logprob` | A | 0.664 [0.562, 0.761] | 0.753 [0.674, 0.830] | 0 | 1.0× |
| `a_perplexity` † | A | 0.664 [0.562, 0.761] | 0.753 [0.674, 0.830] | 0 | 1.0× |
| `a_length_normalized_logprob` | A | 0.662 [0.556, 0.761] | 0.752 [0.664, 0.836] | 0 | 1.0× |
| `a_total_logprob` | A | 0.660 [0.554, 0.758] | 0.742 [0.653, 0.832] | 0 | 1.0× |
| `c_p_true_plain` | C | 0.659 [0.555, 0.755] | **0.756** [0.675, 0.831] | 1 | 2.0× |
| `a_min_logprob` | A | 0.649 [0.543, 0.749] | 0.747 [0.662, 0.828] | 0 | 1.0× |
| `a_max_top5_entropy` | A | 0.636 [0.530, 0.735] | 0.728 [0.643, 0.809] | 0 | 1.0× |
| `a_mean_top5_entropy` | A | 0.633 [0.531, 0.732] | 0.689 [0.599, 0.787] | 0 | 1.0× |
| `c_p_true_with_samples` | C | 0.626 [0.523, 0.725] | 0.713 [0.630, 0.795] | 1 | 2.0× |
| `a_first_token_logprob` | A | 0.580 [0.473, 0.684] | 0.697 [0.613, 0.778] | 0 | 1.0× |
| `t_answer_length` | T | 0.551 [0.475, 0.627] | 0.578 [0.528, 0.643] | 0 | 1.0× |
| `a_first_token_margin` | A | 0.545 [0.439, 0.649] | 0.601 [0.520, 0.717] | 0 | 1.0× |
| `t_random` | T | 0.508 [0.404, 0.611] | 0.542 [0.473, 0.647] | 0 | 1.0× |
| `c_verbal_confidence` | C | 0.484 [0.392, 0.577] | 0.527 [0.487, 0.585] | 1 | 2.0× |

Note that AUROC and AUPRC disagree about the winner: `c_p_true_plain` has the
best AUPRC in the table (0.756) while ranking 12th on AUROC (0.659), and
`a_mean_logprob` is second on AUPRC (0.753) while eighth on AUROC. Which metric
matters depends on your operating point; this benchmark reports both and does
not choose for you.

### Verdict on significance

**No winner.** `b_distinct_count` leads `b_distinct_fraction` by 0.000 AUROC and
their CIs are identical at n=120, because the two are the same signal: they
differ by a constant divisor — the answer-set size, which is `n_samples + 1 = 6`
on every row — and their measured Spearman correlation is exactly +1.000. The
leader's nearest *distinct* competitor is `b_disagreement_rate` at 0.701.

The test used is a **paired stratified bootstrap on the AUROC difference**
(10,000 resamples, identical resample indices for both signals in every
resample), with **Holm–Bonferroni** correction across all 20 comparisons against
the top-ranked signal. This is a documented substitute for DeLong's test for
correlated AUROCs; DeLong's analytic variance assumes untied scores and several
signals here are integer-valued with heavy ties. The substitution is recorded in
`results.json` and `docs/DECISIONS.md`.

Of 20 comparisons, **4 are significant after Holm correction** — all of them
signals that are significantly *worse* than the leader. Three of those 20
comparisons are rank-equivalent duplicates, so the correction is conservative;
recomputing over the 17 distinct comparisons changes no verdict, as noted above.

| signal | Δ AUROC vs leader | p_holm | significant |
|---|---|---|---|
| `a_first_token_logprob` | −0.124 | 0.0288 | yes |
| `t_answer_length` | −0.153 | 0.0408 | yes |
| `a_first_token_margin` | −0.159 | 0.0040 | yes |
| `c_verbal_confidence` | −0.220 | 0.0076 | yes |
| `t_random` | −0.196 | 0.0800 | **no** |
| `t_question_length` | −0.020 | 1.0000 | no |
| `a_mean_logprob` | −0.040 | 1.0000 | no |

No signal is significantly better than any other signal in the top six.
Notably `t_random` is **not** significantly worse than the leader
(p_holm = 0.0800), which is the honest reading of a 0.196 AUROC gap at n=120.
The full 20-row comparison table is in `results.json` under
`views.primary.significance.comparisons`.

### N-ablation: self-consistency saturates at N=3

`b_distinct_count` AUROC against the number of samples used. The first N of the
same five generations are reused at every level, so the differences are due to
sample count and not to drawing a different sample set. A self-check asserts
that N=5 reproduces the main family-B pass; measured maximum difference 0.000.

| N | AUROC [95% CI] |
|---|---|
| 1 | 0.628 [0.553, 0.703] |
| 2 | 0.678 [0.596, 0.754] |
| 3 | **0.705 [0.623, 0.781]** |
| 5 | 0.704 [0.616, 0.786] |

Each extra sample is another full generation, so N=5 costs 6.0× the single
greedy answer (the measured multiplier in `results.json`) against N=3's four
generations for the same discrimination.

Figure: `figures/n_ablation.png`.

### Calibration

ECE on held-out rows only, 10 bins, Platt map fitted on the train split alone
(30% of rows, `train_fraction = 0.3`). Only the three probability-valued signals
have a meaningful pre-Platt ECE — a raw logprob is not a probability, and
squeezing one into [0,1] to score it would measure the squeezing choice rather
than the signal, so the other eighteen store null and are reported as such.

| signal | ECE before Platt | ECE after Platt |
|---|---|---|
| `c_p_true_plain` | 0.200 | 0.146 |
| `c_p_true_with_samples` | 0.330 | 0.127 |
| `c_verbal_confidence` | 0.445 | 0.104 |

Reliability diagrams: **`figures/reliability.png`** — one panel per
probability-valued signal, showing the before-Platt and after-Platt curves
against the diagonal, with both ECE values in each panel's legend. Marker area
is proportional to bin count; empty bins are omitted rather than interpolated
across, because at n=120 over 10 bins several bins hold nothing and a smooth
curve through them would be drawn from data that does not exist.

Recalibration helps all three, and helps `c_verbal_confidence` most
(0.445 → 0.104) — but note that a well-calibrated signal is not a discriminative
one. `c_verbal_confidence` ends with the *best* post-Platt ECE in the table and
an AUROC of 0.484, below chance. Platt scaling is monotone, so it cannot change
AUROC at all; it only moves the numbers onto the right scale.

### Figures

- **`figures/reliability.png`** — reliability diagrams, before and after Platt,
  for the three probability-valued signals (finding 4's calibration counterpart).
- **`figures/risk_coverage.png`** — risk against coverage for the top five
  signals plus `t_random`. The operational picture: how much error you avoid by
  declining to answer the rows a signal flags.
- **`figures/cost_vs_auroc.png`** — AUROC against measured cost with the Pareto
  frontier drawn explicitly. This is finding 2 as a picture.
- **`figures/n_ablation.png`** — family-B AUROC against sample count. This is
  finding 3 as a picture.
- **`figures/auroc.png`** — all 21 signals with bootstrap CIs, grouped by
  family. This is finding 1 as a picture: the intervals visibly overlap.
- **`figures/correlation.png`** — Spearman correlation between every pair of
  signals. Shows how much of a 21-row table is one measurement wearing many
  hats.

Every figure is drawn from `results.json` alone (`make figures`), so a plotting
change never requires rerunning a model and no figure can disagree with the
numbers above.

### Length confound

`b_distinct_count` survives controlling for answer length: 0.678 in the
at-or-below-median-length stratum (n=91) against 0.704 pooled. Spearman
correlation with answer length is 0.44 for `a_length_normalized_logprob` and
0.29–0.30 for the mean-logprob signals, against 0.40 for `b_distinct_count`
and 0.48 for `t_question_length`.

### Signal combination

5-fold CV logistic regression over `b_distinct_count` + `b_disagreement_rate`
(the first partner not rank-identical to the leader): cross-validated AUROC
**0.691** against the best single signal's **0.704**. **The combination does not
beat the best single signal.**

### Verbalized confidence

0 parse failures across 120 rows, 4 distinct values
{0.85: 47, 0.89: 1, 0.95: 9, 1.00: 63}, modal share 0.525 — not effectively
constant, so it is a real signal and not a degenerate one. It scores AUROC
**0.484**, below chance. On a valid base rate with orientation unit-tested, this
is a legitimate result: a 0.5B model's stated confidence is anti-informative
about whether it is right.

## Reproduction

```bash
uv sync --extra local
uv run unc-bench build-dataset  --config configs/run2.yaml
uv run unc-bench generate       --config configs/run2.yaml
uv run unc-bench score-signals  --config configs/run2.yaml --family b
uv run unc-bench score-signals  --config configs/run2.yaml --family actc
uv run unc-bench ablation       --config configs/run2.yaml
uv run unc-bench label          --config configs/run2.yaml   # needs GSK_API_KEY
uv run unc-bench analyze        --config configs/run2.yaml
uv run unc-bench figures        --config configs/run2.yaml
```

Family B must run as its own pass: the NLI model and the generator do not fit in
2 GB together.

### Run #3: the same study at n=600 on a GPU

Every number in this README is run #2: Qwen2.5-0.5B-Instruct, 2 CPU cores, ~2 GB
RAM, n=120, PopQA 90 / TriviaQA 30. That configuration is what the hardware
allowed, and two of its limitations are consequences of the hardware rather than
of the design — the small n, and the 90/30 split that is the mechanism behind
finding 4 and behind the pooled table's distortion (limitation 13).

`configs/run3_gpu.yaml` with `notebooks/run_on_colab.ipynb` runs the same
pipeline at n=600 with a balanced 300 PopQA / 300 TriviaQA split, on
Qwen2.5-3B-Instruct in fp16 on a free Colab T4. The config sets no quantization:
quantization distorts the token logprobs that signal family A reads. It also sets
`generation_batch_size: 1`, and that value must not be raised — open defect D27
(`docs/DECISIONS.md`) records that padding a ragged batch perturbs per-token
logprobs by up to 2.52e-02 while a uniform-length batch is bit-identical, and
D27 is unfixed. Run #3 also turns on the per-dataset bootstrap intervals, which
are written and tested but have never had data: run #2's per-row values were lost
before they could be committed, which is why run #2's per-dataset table falls
back to a Hanley–McNeil normal approximation.

The notebook's first cell estimates the run at **roughly 1.5 to 2.5 hours** and
shows the arithmetic behind that band. It is an estimate, not a measurement.

**Run #3 has not been run.** No run #3 number appears anywhere in this
repository, and none of run #2's findings above is weakened or withdrawn in
anticipation of it. Run #2 stands as published, at n=120, with the limitations
it has.

### Determinism

Stated precisely, because the two halves of the pipeline have different
evidence behind them:

- **Analysis is reproducible byte-for-byte. Measured.** Re-running `analyze` on
  the same artifacts produced a `results.json` **byte-identical** to the
  published one apart from the timestamp field. This was checked, not assumed.
- **Generation reproducibility is unmeasured.** Whether generation on a clean
  clone is bit-exact was never tested. Decoding is greedy at temperature 0 with
  seed 0, which removes sampling as a source of variation, but CPU
  floating-point kernels are not guaranteed identical across `transformers` or
  `torch` builds and no re-run was performed to check. The repository ships a
  `nondeterminism` stage (`make nondeterminism`) that would quantify this; it
  has not been run against these rows. **No claim is made either way.**

All seeds are recorded in `results.json` under `seeds`: greedy 0, sampling base
1000, dataset 12345, split 20260101, logreg 31337, bootstrap 987654321.

## Hardware and wall clock

2 CPU cores, ~2 GB RAM, no GPU (`Linux-6.1.155-x86_64`, Python 3.11.16).
Generation measured **16.8–19.0 s/question** (4.9 s/item on the final cache-warm
chunk). Family B including NLI clustering **0.52 s/item**. Total session wall
clock, including two pilot iterations and three analysis re-runs, roughly
**5 hours**.

## Model card

Qwen2.5-0.5B-Instruct, bfloat16, local `transformers` 4.46.3 / torch 2.5.1.
Greedy decoding: temperature 0, top_p 1.0, seed 0, max_new_tokens 24. Family B
sampling: 5 draws at temperature 0.7, top_p 0.9, seed base 1000. NLI for
bidirectional-entailment clustering: `MoritzLaurer/DeBERTa-v3-base-mnli`.

## Dataset card

120 questions: **90 PopQA** restricted to the top popularity decile *and* to five
lookup-style Wikidata relations (`capital`, `country`, `capital of`, `sport`,
`color`); **30 TriviaQA** restricted to high-alias-count (≥20 aliases) short
(≤20 word) questions. TriviaQA ships no popularity field, so alias count and
question length are a **documented proxy substitution**, not a popularity
measurement. A 2-shot exemplar prefix is asserted in code to be disjoint from
the eval set. The two subsets have very different base rates (40% vs 90%
incorrect), which is the whole reason finding 4 matters.

## Inter-judge agreement

Cohen's **κ = 0.849** on **n = 66**, observed agreement 0.985. The denominator is
the full overlap: 66 rows were sent to both `gpt-5-mini` and `claude-haiku-4-5`,
66 parsed, **0 dropped for parse failure**, and the equality is asserted in code
so a κ can never be quoted over a smaller row set than it claims. The other 54
of 120 rows were settled by exact match and never sent to a judge.
`data/human_validation_sample.csv` holds 100 rows with an **empty** `human_label`
column; no human has labeled it.

## What I would ship

**Nothing, on this evidence.** This benchmark does not support shipping any of
these 21 signals at this scale.

An earlier version of this section recommended `a_mean_logprob` at the 40th
percentile. That recommendation was inconsistent with this README's own primary
table and has been replaced; the reasoning is recorded in
[docs/DECISIONS.md](docs/DECISIONS.md) rather than quietly deleted.

### Why the pooled ranking is not the basis

The superseded recommendation rested on pooled AUROC 0.664 and the pooled AUPRC
0.753. Both are pooled numbers, and this README designates the
[per-dataset table](#per-dataset-auroc) as primary for a reason it also states:
`t_random` — a random number, which cannot predict anything — scores 0.691 within
TriviaQA. That column is estimator noise from 3 correct rows in 30, and the
README says it should not be read as a ranking.

`a_mean_logprob` scores **0.753 on TriviaQA and 0.514 on PopQA**. Its pooled
0.664 is substantially earned in the column the README disowns. Recommending it
on a pooled number while disowning the subset that produces that number is the
inconsistency this section fixes. The same objection voids the AUPRC argument:
pooled AUPRC is computed over the same mixed set and inherits the same defect,
and `a_mean_logprob` is not even the AUPRC leader — `c_p_true_plain` is, at
0.756.

### What the primary table supports

PopQA is the subset worth reading: n=90, 40% incorrect, 36 positives against 54
negatives. The per-dataset block in `results.json` stores point estimates with no
interval, and the per-row signal values needed to bootstrap one are not in the
repository (`.gitignore` excludes the signal artifacts). So the interval below is
a **Hanley–McNeil normal approximation**, computable from the stored AUROC and
class counts alone. It is *not* the percentile bootstrap used everywhere else in
this README, and it is labelled as an approximation wherever it appears.
Calibration: across the 21 pooled signals, where both methods can be compared on
the same rows, the two disagree by at most **0.0267** on an interval endpoint and
never disagree about whether 0.50 is excluded.

| signal | PopQA AUROC | analytic 95% interval | excludes 0.50 |
|---|---|---|---|
| `c_p_true_plain` | 0.626 | [0.507, 0.746] | yes, by 0.007 |
| `b_distinct_count` | 0.603 | [0.482, 0.724] | no |
| `b_disagreement_rate` | 0.601 | [0.480, 0.722] | no |
| `b_mean_pairwise_f1` | 0.595 | [0.474, 0.716] | no |
| `b_semantic_entropy` | 0.579 | [0.458, 0.701] | no |
| `a_mean_logprob` | 0.514 | [0.392, 0.637] | no |
| `t_random` | 0.511 | [0.388, 0.633] | no |
| `t_question_length` | 0.505 | [0.383, 0.628] | no |
| `c_verbal_confidence` | 0.395 | [0.274, 0.516] | no |

All 21 rows: `unc-bench audit`.

**Exactly one signal's PopQA interval excludes 0.50, and that exclusion does not
survive scrutiny.** `c_p_true_plain` clears chance by 0.007 on the lower
endpoint. That margin is smaller than the 0.0267 disagreement between this
approximation and the bootstrap the project actually reports, so the exclusion is
an artifact of the interval method as easily as a property of the data. It is
also a selected maximum: it is the largest of 18 distinct signals scanned on the
same subset. Its unadjusted p against 0.50 is 0.0385; Bonferroni over 18
orderings gives 0.693. It does not clear chance once the selection is accounted
for.

**So the honest answer is that no signal clears chance on PopQA.** I record this
as inconclusive rather than as a negative result: an exact bootstrap interval on
the PopQA subset is not computable from the committed artifacts, so "no signal is
established above chance" is what the evidence supports, not "no signal works".

### What would change the answer

- **Larger n.** At 36 positives the analytic interval half-width is about 0.12,
  so a true AUROC of 0.63 cannot be separated from 0.50. Separating 0.60 from
  0.50 at this balance needs roughly 250–300 rows per subset.
- **A larger subject model.** Qwen2.5-0.5B-Instruct is a substitute forced by
  hardware (docs/DECISIONS.md D1–D3). Its errors are dominated by not knowing the
  fact at all, which is the regime where confidence signals have least to say.
- **A harder, more balanced dataset.** The 90/30 split with base rates of 40% and
  90% is the mechanism behind finding 4; see limitation 13. Equal subset sizes at
  similar base rates would let the per-dataset columns be compared directly
  instead of one being written off as noise.
- **Committing the signal columns.** Then per-dataset bootstrap intervals could
  be computed with the same estimator as the pooled table, and this section would
  not depend on an approximation.

### The one thing the table does support

If self-consistency is used anyway, use **N=3, not N=5**: AUROC saturates at
0.705 by N=3 and does not improve at N=5, so two fifths of family B's 6× cost
buys nothing measurable. That claim is about a cost–benefit ratio *within* a
signal, not about whether the signal beats chance, so the objection above does
not apply to it.

## Limitations

Two that bear directly on how the numbers above should be read:

- **No label is human-verified.** The inter-judge κ of 0.849 measures how
  consistently two judges agree with *each other*, not whether either agrees with
  a human, and no human has verified any label in this run;
  `data/human_validation_sample.csv` ships with its `human_label` column empty
  and `unc-bench human-agreement` is wired up and tested but has never been run
  on real labels. Once the column is filled, the comparison is:

  ```bash
  uv run unc-bench human-agreement --csv data/human_validation_sample.csv
  ```

  It reports Cohen's κ between the human column and the machine label, and the
  per-source breakdown. It runs today against the empty column and reports zero
  usable rows, which is the correct answer for a template.
- **The 90/30 dataset split is a design flaw**, not just an inconvenience. It is
  the mechanism behind finding 4 and behind the pooled table's distortion. See
  limitation 13.

Full list: [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Decision log:
[docs/DECISIONS.md](docs/DECISIONS.md).

## Run #1 and why it was discarded

Run #1 (tag [`run1-n100`](../../tree/run1-n100)) is preserved and its numbers are
not used anywhere above.

It answered 75 of 100 questions and got **7 right — a 90.7% error rate** — with 25
abstentions. With 7 positives in the minority class, the random baseline
`t_random` scored **AUROC 0.746** instead of the ~0.50 it must score by
construction. A random number cannot predict anything, so a random signal at
0.746 is proof that the estimator, not the signals, was generating the ordering.
At 7 positives one row flipping moves AUROC by about 0.14. The entire 21-signal
ranking was noise.

The fix was to change the task, not the model: an easier question mix, no
abstention instruction, and a 2-shot prefix. That moved the base rate from 9.3%
correct to 50.0% on the heuristic labeler — **57 correct / 63 incorrect = 47.5%
after judging**, which is the figure every number above is computed on — and the
abstention rate from 0.25 to 0.000. The validity gates
that would have caught run #1 are now code assertions in the analysis stage, and
they run on every analysis.

The same failure mode is still visible *inside* this run's TriviaQA subset,
where 3 correct rows out of 30 push `t_random` to 0.691 — which is why the
per-dataset table above is presented with that warning attached rather than as a
second ranking.

## License

MIT. See [LICENSE](LICENSE).
