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
["What I would ship"](#what-i-would-ship): ship `a_mean_logprob`, which is free,
and do not ship self-consistency.

## Per-dataset AUROC

The primary table. Run #2, n=120, `configs/run2.yaml`, positive class
**incorrect answer**. Two datasets with very different base rates, scored
separately so no signal can earn credit for telling them apart.

| signal | family | cost | PopQA AUROC (n=90, 40% incorrect) | TriviaQA AUROC (n=30, 90% incorrect) | pooled AUROC |
|---|---|---|---|---|---|
| `b_distinct_count` | B | 6.0× | 0.603 | 0.741 | 0.704 |
| `b_distinct_fraction` | B | 6.0× | 0.603 | 0.741 | 0.704 |
| `b_disagreement_rate` | B | 6.0× | 0.601 | 0.685 | 0.701 |
| `b_semantic_entropy` | B | 6.0× | 0.579 | 0.704 | 0.693 |
| `b_semantic_entropy_normalized` | B | 6.0× | 0.579 | 0.704 | 0.693 |
| `b_mean_pairwise_f1` | B | 6.0× | 0.595 | 0.691 | 0.691 |
| `t_question_length` | T | 1.0× | 0.505 | 0.580 | 0.684 |
| `a_mean_logprob` | A | 1.0× | 0.514 | 0.753 | 0.664 |
| `c_p_true_plain` | C | 2.0× | **0.626** | 0.660 | 0.659 |
| `t_random` | T | 1.0× | 0.511 | 0.691 | 0.508 |
| `c_verbal_confidence` | C | 2.0× | 0.395 | **0.765** | 0.484 |

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

| signal | family | AUROC [95% CI] | AUPRC [95% CI] | extra calls/q | cost |
|---|---|---|---|---|---|
| `b_distinct_count` | B | 0.704 [0.616, 0.786] | 0.724 [0.651, 0.796] | 5 | 6.0× |
| `b_distinct_fraction` | B | 0.704 [0.616, 0.786] | 0.724 [0.651, 0.796] | 5 | 6.0× |
| `b_disagreement_rate` | B | 0.701 [0.612, 0.783] | 0.716 [0.636, 0.795] | 5 | 6.0× |
| `b_semantic_entropy` | B | 0.693 [0.604, 0.774] | 0.712 [0.635, 0.789] | 5 | 6.0× |
| `b_semantic_entropy_normalized` | B | 0.693 [0.604, 0.774] | 0.712 [0.635, 0.789] | 5 | 6.0× |
| `b_mean_pairwise_f1` | B | 0.691 [0.600, 0.775] | 0.714 [0.635, 0.794] | 5 | 6.0× |
| `t_question_length` | T | 0.684 [0.599, 0.763] | 0.697 [0.618, 0.788] | 0 | 1.0× |
| `a_mean_logprob` | A | 0.664 [0.562, 0.761] | 0.753 [0.674, 0.830] | 0 | 1.0× |
| `a_perplexity` | A | 0.664 [0.562, 0.761] | 0.753 [0.674, 0.830] | 0 | 1.0× |
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
their CIs overlap at n=120. The two are rank-identical by construction (they
differ by a constant divisor of 5).

The test used is a **paired stratified bootstrap on the AUROC difference**
(10,000 resamples, identical resample indices for both signals in every
resample), with **Holm–Bonferroni** correction across all 20 comparisons against
the top-ranked signal. This is a documented substitute for DeLong's test for
correlated AUROCs; DeLong's analytic variance assumes untied scores and several
signals here are integer-valued with heavy ties. The substitution is recorded in
`results.json` and `docs/DECISIONS.md`.

Of 20 comparisons, **4 are significant after Holm correction** — all of them
signals that are significantly *worse* than the leader:

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

I would ship **`a_mean_logprob`** with a threshold at the 40th percentile of its
oriented score, and I would not ship self-consistency.

The reasoning is cost, not accuracy. Self-consistency leads on AUROC (0.704 vs
0.664) but costs 6× and its CI [0.616, 0.786] overlaps mean-logprob's
[0.562, 0.761] completely, so the gap is not established at n=120. Mean logprob
is free — the logprobs arrive with the answer that was already generated — and it
has the *second-best AUPRC in the whole table* (0.753 vs the leader's 0.724),
which is the metric that matters when you are trying to catch errors in a stream
where errors are the minority. Paying 6× for an unestablished 0.04 AUROC gain is
not a trade I would make. If the ablation is to be believed, anyone who does want
self-consistency should use **N=3, not N=5**: AUROC saturates at 0.705 by N=3 and
does not improve at N=5, so two fifths of family B's cost buys nothing.

The two honest caveats on that recommendation are findings 2 and 4.
`t_question_length` scores 0.684 at the same 1× cost, which means a chunk of what
mean-logprob "detects" is just question difficulty; and within PopQA alone
mean-logprob falls to 0.514 — essentially chance — while the leader falls to
0.603, so the pooled numbers are flattered by the two datasets' different base
rates. On a single-distribution deployment I would expect worse than 0.664 from
any signal here, and I would re-measure rather than trust the pooled number.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). Decision log:
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
