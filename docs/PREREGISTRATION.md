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
abstention rate below 0.10; and the two-phase human-label rule, whose order
matters (see below): PRE-run `labeling_protocol_validated` requires the
protocol sample (`data/human_validation_sample.csv`) at ≥0.50 coverage, and
POST-run `human_label_coverage` requires the new run's own validation
subsample at ≥0.80 coverage. Both gates are defined in
`src/unc_bench/analysis/validity.py`, tracked in `docs/OPEN_DEFECTS.md`, and
named here so the three cannot disagree (pinned by `scripts/audit_docs.py`).

## Ordering (read before executing anything)

1. Label the prior sample (`data/human_validation_sample.csv`) per
   `docs/HUMAN_LABELING.md` until coverage ≥ 0.50. This validates the
   instructions against real edge cases and opens the PRE-run gate.
2. Run the pilot (build → generate → label → pilot-gate). If the base rate
   lands outside 25–65%, the run is a record of that failure; re-mixing ends
   after the two permitted iterations.
3. Execute the full run (generate → score → ablation → label → analyze).
4. Generate the new run's validation subsample (100 rows, balanced on machine
   label, `human_label` empty) and label it to ≥ 0.80 coverage.
5. Re-run `analyze` so `human_label_coverage` and `label_quality` reflect the
   labels. Only now, with all gates passing, is the ranking publishable.

Skipping step 1 makes step 4's gate vacuous (labels under untested
instructions); skipping step 4 leaves the POST-run gate failing and the
ranking unpublished. The order is the control.

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
