# Pre-registration: run #3 (n=600, balanced 300/300, Qwen2.5-3B-Instruct fp16)

Committed BEFORE run #3 is executed. No run #3 number exists anywhere in this
repository; when the run lands, this document decides what counts as a finding
and what counts as noise. If the run contradicts this document, the document
wins the argument about what was predicted, and the run wins the argument
about what is true.

## Primary question

Does self-consistency (family B) beat single-pass logprobs (family A) at
predicting that the model answered wrong — and is either of them worth more
than the trivial baselines (family T)?

## Primary metric and subset

- Metric: AUROC for the INCORRECT positive class, with the stratified
  percentile bootstrap interval (resampled within class; cluster-aware per
  `analysis.cluster_bootstrap`).
- Subset: PopQA alone (n=300). PopQA is the larger, more balancable subset;
  the pooled number is disqualified as primary because any signal correlated
  with dataset provenance earns pooled AUROC for free (finding 4). The
  stratified pooled AUROC (`unc-bench audit`) is reported beside it, not
  instead of it.

## Exact comparisons (in order)

1. Best family-B signal vs best family-A signal, paired bootstrap on the AUROC
   difference, Holm-corrected over the distinct-signal family.
2. Winner of (1) vs `t_question_length` at 1× cost: a 6× signal that cannot
   separate from a word count has no deployment case.
3. `c_p_true_plain` vs chance on PopQA: run #2's only chance-clearing PopQA
   interval (0.626 [0.507, 0.746], analytic) with a margin smaller than the
   method disagreement — this run retests it with a real bootstrap interval.

## Validity gates (all must pass or the ranking is not publishable)

Random-baseline CI contains 0.50; ≥30 rows per class in each subset;
abstention rate below 0.10; human-label coverage ≥0.80
(`analysis.validity.human_label_coverage` — currently failing at 0.0, which
means this pre-registration ALSO requires the labelling in
`docs/HUMAN_LABELING.md` to happen before the run counts).

## Stopping rule

One run at the committed config. No re-filtering, no seed-hunting, no
subset re-weighting after seeing labels. If the pilot gate fails
(35–65% band), the run is a record of that failure, not an invitation to a
third pilot: the gate permits two iterations and both are budgeted.

## Power

Hanley–McNeil analytic interval at 150 positives / 150 negatives, AUROC 0.60:
[0.536, 0.664], half-width 0.064 — computed from the project's own
`analytic_auroc_ci`, not guessed. A true AUROC of 0.60 separates from 0.50 on
a balanced 300-row subset; run #2's 36/54 PopQA split could not (half-width
0.120). This is an approximation (it assumes no ties); the run reports
bootstrap intervals and this number only sizes the study.

## What falsifies the hypothesis

"Self-consistency beats logprobs" is falsified if, on PopQA, the best family-B
signal is not significantly above the best family-A signal after Holm — OR if
neither clears chance, in which case the finding is "no signal established",
not "the cheaper one wins". A ranking inversion across model scale (0.5B run
#2 vs 3B run #3 on comparable questions) would be reported as the headline,
not buried: it directly answers LIMITATIONS item 2.
