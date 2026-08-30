# Which cheap uncertainty signal best predicts an LLM's factual errors?

Twenty-one uncertainty signals from three families — token logprobs (1× cost),
self-consistency over 5 samples (6× cost), and self-verification / P(True)
(2× cost) — ranked by how well each predicts that Qwen2.5-0.5B-Instruct got a
short factual question wrong.

## Validity gates

**ALL THREE GATES PASS.** Random baseline `t_random` AUROC **0.508 [0.404, 0.611]**
(CI contains 0.50, as it must). **63 incorrect / 57 correct** (both classes ≥ 30).
Abstention rate **0/120 = 0.000** (below the 0.10 ceiling). The ranking below is
publishable. Run #1's ranking was not; see the last section.

## Results

Run #2, n=120, `configs/run2.yaml`. Every number below was computed in this run
on the identical 120-row frozen analysis set (`qid_digest = ffff86216137caed`).

### AUROC and AUPRC

Positive class is **incorrect answer**. Higher is better. AUPRC baseline (the
prevalence of the positive class) is **0.525** — an AUPRC at or below that
number is no better than guessing.

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
| `c_p_true_plain` | C | 0.659 [0.555, 0.755] | 0.756 [0.675, 0.831] | 1 | 2.0× |
| `a_min_logprob` | A | 0.649 [0.543, 0.749] | 0.747 [0.662, 0.828] | 0 | 1.0× |
| `a_max_top5_entropy` | A | 0.636 [0.530, 0.735] | 0.728 [0.643, 0.809] | 0 | 1.0× |
| `a_mean_top5_entropy` | A | 0.633 [0.531, 0.732] | 0.689 [0.599, 0.787] | 0 | 1.0× |
| `c_p_true_with_samples` | C | 0.626 [0.523, 0.725] | 0.713 [0.630, 0.795] | 1 | 2.0× |
| `a_first_token_logprob` | A | 0.580 [0.473, 0.684] | 0.697 [0.613, 0.778] | 0 | 1.0× |
| `t_answer_length` | T | 0.551 [0.475, 0.627] | 0.578 [0.528, 0.643] | 0 | 1.0× |
| `a_first_token_margin` | A | 0.545 [0.439, 0.649] | 0.601 [0.520, 0.717] | 0 | 1.0× |
| `t_random` | T | 0.508 [0.404, 0.611] | 0.542 [0.473, 0.647] | 0 | 1.0× |
| `c_verbal_confidence` | C | 0.484 [0.392, 0.577] | 0.527 [0.487, 0.585] | 1 | 2.0× |

### Significance

| `a_first_token_logprob` | -0.124 | 0.0288 |
| `t_answer_length` | -0.153 | 0.0408 |
| `a_first_token_margin` | -0.159 | 0.0040 |
| `c_verbal_confidence` | -0.220 | 0.0076 |

### N-ablation

| 1 | 0.628 [0.553, 0.703] |
| 2 | 0.678 [0.596, 0.754] |
| 3 | 0.705 [0.623, 0.781] |
| 5 | 0.704 [0.616, 0.786] |

### Calibration

| `c_p_true_plain` | 0.200 | 0.146 |
| `c_p_true_with_samples` | 0.330 | 0.127 |
| `c_verbal_confidence` | 0.445 | 0.104 |

### Per-dataset

| popqa | 90 | 0.40 | 0.603 | 0.626 |
| triviaqa | 30 | 0.90 | 0.741 | 0.765 |

### Verdict on significance

**No winner.** `b_distinct_count` leads `b_distinct_fraction` by 0.000 AUROC and
their CIs overlap at n=120. The two are rank-identical by construction (they
differ by a constant divisor of 5).

The test used is a **paired stratified bootstrap on the AUROC difference**
(10,000 resamples, identical resample indices for both signals in every
resample), with **Holm–Bonferroni** correction across all 20 comparisons against
the top-ranked signal. This is a documented substitute for DeLong's test for
correlated AUROCs; the substitution was made on runtime grounds and is recorded
in `results.json` and `docs/DECISIONS.md`.

Of 20 comparisons, **4 are significant after Holm correction** — all of them
signals that are significantly *worse* than the leader. No signal is
significantly better than any other signal in the top six. Notably `t_random` is
**not** significantly worse than the leader (p_holm = 0.0800), which is the
honest reading of a 0.196 AUROC gap at n=120.

### Figures

Risk–coverage: `figures/risk_coverage.png`. N-ablation: `figures/n_ablation.png`.
Cost vs AUROC with Pareto frontier: `figures/cost_vs_auroc.png`. Also
`figures/auroc.png`, `figures/calibration.png`, `figures/correlation.png`.

### Length confound

`b_distinct_count` survives controlling for answer length: 0.678 in the
at-or-below-median-length stratum (n=91) against 0.704 pooled. Spearman
correlation with answer length is 0.44 for `a_length_normalized_logprob` and
0.29–0.30 for the mean-logprob signals, so family A is more length-entangled
than family B.

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

**Measured, not assumed.** Re-running `analyze` on the same artifacts produced a
`results.json` **byte-identical** to the published one apart from the timestamp
field. All seeds are recorded in `results.json` under `seeds`: greedy 0, sampling
base 1000, dataset 12345, split 20260101, logreg 31337, bootstrap 987654321.
Generation on a clean clone is greedy at temperature 0, but CPU floating-point
kernels are not guaranteed identical across transformers builds, so generation is
reproducible in practice and not guaranteed bit-exact.

## Hardware and wall clock

2 CPU cores, ~2 GB RAM, no GPU. Generation measured **16.8–19.0 s/question**
(4.9 s/item on the final cache-warm chunk). Family B including NLI clustering
**0.52 s/item**. Total session wall clock, including two pilot iterations and
three analysis re-runs, roughly **5 hours**.

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
the eval set.

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

The two honest caveats on that recommendation: `t_question_length` scores 0.684 at
the same 1× cost, which means a chunk of what mean-logprob "detects" is just
question difficulty; and within PopQA alone the leader falls to 0.603, so the
pooled numbers are flattered by the two datasets' different base rates.

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
correct to 50.0% and the abstention rate from 0.25 to 0.000. The validity gates
that would have caught run #1 are now code assertions in the analysis stage, and
they run on every analysis.
