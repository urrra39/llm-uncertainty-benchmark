# Limitations

Written against run #2 (n=120, `configs/run2.yaml`). Run #1 is discarded; see
"Run #1 and why it was discarded" in the README.

1. **n=120 is small.** AUROC confidence intervals are roughly ±0.09 wide. The
   top six signals are statistically indistinguishable from each other. The
   benchmark can separate "clearly better than chance" from "not", and cannot
   separate the leaders from each other. Any claim that one family B signal
   beats another is unsupported at this n.

2. **One model, one size.** Qwen2.5-0.5B-Instruct on CPU. Nothing here
   generalizes to larger models. A 0.5B model's logprobs may be worse calibrated
   than a 7B model's, which would make family A look worse than it deserves.

3. **The base rate landed in range partly by luck.** The 40-row pilot projected
   36.6% correct; the full run measured 50.0% on the heuristic labeler and 47.5%
   (57/120) after judging. The projection was 13 points off.
   The gate (35–65%) was not met by either pilot iteration; the full run met the
   40–60% target anyway. See docs/DECISIONS.md.

4. **The dataset is deliberately easy and therefore unrepresentative.** PopQA is
   restricted to five lookup-style relations (`capital`, `country`, `capital of`,
   `sport`, `color`) and TriviaQA to high-alias-count short questions. This was
   necessary to get a measurable base rate out of a 0.5B model, and it means the
   question distribution is not PopQA's or TriviaQA's.

5. **A cheap non-signal is nearly competitive.** `t_question_length` scores 0.684
   at 1× cost against the leader's 0.704 at 6× cost, their CIs overlap, and the
   paired bootstrap puts the difference at p_holm = 1.0. Question length is a
   property of the *input*, not of the model's uncertainty. Its strength means a
   meaningful fraction of what the uncertainty signals detect is "this question
   is hard", which any keyword heuristic could also detect. This is finding 2 in
   the README.

6. **The pooled ranking is inflated by the dataset mix.** `b_distinct_count`
   scores 0.704 pooled but 0.603 within PopQA (n=90) and 0.741 within TriviaQA
   (n=30). The two datasets have very different base rates (0.40 vs 0.90
   incorrect), so a signal that merely separates datasets gains pooled AUROC.
   The README leads with the per-dataset table for this reason (finding 4).

   The TriviaQA column is itself weak evidence and should not be read as a
   second ranking: with only 3 correct rows in 30, `t_random` scores 0.691 there
   — the same estimator-noise failure that invalidated run #1, at a smaller
   scale.

7. **Labels are model-judged, not human-verified.** 54 of 120 rows settled by
   exact match; 66 by two LLM judges at κ=0.849. `data/human_validation_sample.csv`
   ships with 100 rows and an **empty** `human_label` column. No human has
   labeled it.

8. **Cost is measured on this hardware only.** 2 CPU cores, no GPU. The 6×
   multiplier for family B is a token/call count; the wall-clock ratio on a GPU
   with batching would be much lower, which would change the cost-vs-AUROC
   conclusion.

9. **The N-ablation reuses one set of five samples.** N=1 means "the first of
   the five", not "an independent single-sample run". Variance across which
   sample is drawn is not measured.

10. **AUPRC ranking disagrees with AUROC ranking.** `c_p_true_plain` has the
    best AUPRC (0.756) while ranking 12th on AUROC (0.659). Both are reported.
    Which matters depends on the operating point, and this benchmark does not
    choose one for you.

11. **Generation reproducibility is unmeasured.** The analysis stage was checked
    and is byte-identical on re-run apart from the timestamp. Whether generation
    on a clean clone is bit-exact was never tested. Decoding is greedy at
    temperature 0 with seed 0, which removes sampling as a source of variation,
    but CPU floating-point kernels are not guaranteed identical across
    `transformers` or `torch` builds. The `nondeterminism` stage that would
    quantify this exists in the repository and has not been run against these
    rows. No claim is made in either direction; the README says the same.

12. **A well-calibrated signal here is not a discriminative one.** Platt scaling
    cuts `c_verbal_confidence`'s ECE from 0.445 to 0.104 — the best post-Platt
    ECE of the three probability-valued signals — while its AUROC stays at 0.484,
    below chance. Platt is monotone and cannot change AUROC. The reliability
    diagrams (`figures/reliability.png`) should not be read as a ranking of
    usefulness.

13. **The 90/30 dataset split is a design flaw, and it is the mechanism behind
    finding 4.** The run pools a 90-row PopQA subset at 40% incorrect with a
    30-row TriviaQA subset at 90% incorrect. Two things follow, and both show up
    in the tables above.

    Any signal correlating with *which dataset a row came from* earns pooled
    AUROC for free, because the datasets differ in base rate. That is exactly
    what `t_question_length` does: 0.505 within PopQA, 0.580 within TriviaQA,
    0.684 pooled. It is not detecting error, it is detecting provenance.

    And the 30-row subset cannot support an estimate. With 3 correct rows,
    `t_random` scores 0.691 there — a random number, which cannot predict
    anything. One row flipping moves that column by roughly 0.1 AUROC. The
    TriviaQA column is noise, which is why the README declines to read it as a
    ranking; but the pooled column that *is* read as a ranking is 25% built from
    it.

    A balanced design would have been roughly 60/60 with base rates within about
    10 points of each other — close enough that pooling adds no dataset signal,
    and large enough per subset that each column carries its own interval. A
    future run should target equal subset sizes and matched base rates, and should
    treat the per-dataset columns as the primary result from the start rather than
    promoting them after the fact.

    The 90/30 weighting was deliberate but was chosen for a different objective:
    the pilot measured PopQA at 41.7% correct against TriviaQA's 21.4%, so the
    mix was weighted toward PopQA to lift the pooled base rate into the gate's
    35–65% band. It was chosen to fix the base rate, not to balance the subsets,
    and the interaction with per-dataset estimation was not considered. See
    `docs/DECISIONS.md`.

14. **No human has verified any label, and the reported κ does not measure
    correctness.** Every one of the 120 labels is machine-assigned: 54 by
    normalized exact match, 66 by an LLM judge. The κ of 0.849 in `results.json`
    is judge-versus-judge over the 66 judged rows — it measures whether
    `gpt-5-mini` and `claude-haiku-4-5` agree with each other, which is judge
    *consistency*. Two judges sharing the same blind spot agree perfectly and are
    both wrong, and κ cannot detect that.

    `data/human_validation_sample.csv` holds a 100-row sample laid out for a
    human, with an empty `human_label` column, and `unc-bench human-agreement`
    computes agreement and Cohen's κ against a filled-in column. It is wired into
    the CLI and tested; it has never been run on real labels, because there are
    none. The column is empty on purpose — filling it without a human would be
    fabrication. Until it is filled, the label set's *correctness* is unmeasured
    in both directions.

    There is at least one row where this matters. For "What is Stockholm the
    capital of?" the gold list contains both "Stockholm County" and "Sweden"; the
    model answered "Stockholm"; the containment heuristic scored that correct and
    the judge scored it incorrect. Which is right is a human judgement that has
    not been made. 9 of the 100 sampled rows have the heuristic and the judge
    disagreeing.
