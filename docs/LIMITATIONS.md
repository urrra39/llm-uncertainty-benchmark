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
   36.6% correct; the full run measured 50.0%. The projection was 13 points off.
   The gate (35–65%) was not met by either pilot iteration; the full run met the
   40–60% target anyway. See docs/DECISIONS.md.

4. **The dataset is deliberately easy and therefore unrepresentative.** PopQA is
   restricted to five lookup-style relations (`capital`, `country`, `capital of`,
   `sport`, `color`) and TriviaQA to high-alias-count short questions. This was
   necessary to get a measurable base rate out of a 0.5B model, and it means the
   question distribution is not PopQA's or TriviaQA's.

5. **A cheap non-signal is nearly competitive.** `t_question_length` scores 0.684
   at 1× cost against the leader's 0.704 at 6× cost, and its CI overlaps the
   leader's. Question length is a property of the *input*, not of the model's
   uncertainty. Its strength means a meaningful fraction of what the uncertainty
   signals detect is "this question is hard", which any keyword heuristic could
   also detect.

6. **The pooled ranking is inflated by the dataset mix.** `b_distinct_count`
   scores 0.704 pooled but 0.603 within PopQA (n=90) and 0.741 within TriviaQA
   (n=30). The two datasets have very different base rates (0.40 vs 0.90
   incorrect), so a signal that merely separates datasets gains pooled AUROC.
   The per-dataset table in the README is the more honest read.

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
