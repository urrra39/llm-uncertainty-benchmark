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

---

# Pre-registration: run #2b (n=120, balanced 60/60, Qwen2.5-0.5B-Instruct CPU)

Committed BEFORE run #2b is executed. `configs/run2b_clean.yaml` differs from
`configs/run2.yaml` only in dataset construction (no `capital of`, leakage
rows dropped, near-duplicates deduped, 60/60 split) plus the infrastructural
improvements that postdate run #2 (per-question sample seeds, exhaustive
primary clusterer with audit, cluster bootstrap, distinct-signal Holm,
signal coverage, token prices). Decoding, prompts, sampling counts, NLI model
and all seeds are identical.

## Power, stated first

Hanley–McNeil analytic interval at 30/30, AUROC 0.60: [0.456, 0.744],
half-width 0.145 (computed, not guessed). A true 0.60 does NOT separate from
0.50 at 60 rows per subset. Run #2b is therefore powered to detect only large
effects and is primarily a **validity** run: its job is to show that
decontamination changes the labels in the predicted direction, not to rank
18 (+6 samples-only) signals.

## Primary metric and subset

AUROC for INCORRECT with stratified percentile bootstrap intervals, on PopQA
alone (n=60) and on the stratified pooled mean. The pooled raw AUROC is
reported for continuity and must not be quoted as a finding (finding 4).

## The falsifiable prediction (E4)

The contamination hypothesis predicts that on clean labels, the signals that
sat at or below chance in run #2's confident stratum move UP:
`c_verbal_confidence` (0.484 pooled) and `a_mean_logprob` (0.514 on PopQA).
The mechanism being removed — near-random labels concentrated exactly where
the signals say "fine" — can only have pushed those numbers down.

Falsification: if they do not move up, the contamination hypothesis is wrong
or incomplete, and the README headline becomes that — not a rescued ranking.
Confound, stated in advance: run #2b labels are heuristic (exact match +
containment; no judge credentials in this environment), while run #2's were
judged. Containment itself scores echo-shaped answers correct when the subject
sits inside an alias, so part of any movement may be labeler change rather
than decontamination. The comparison is valid only as a direction check, and
is labelled as such everywhere it appears.

## Gates and stopping

Same five gates (random-baseline CI, ≥30 per class per subset where
applicable, abstention < 0.10, protocol ≥ 0.50, run coverage ≥ 0.80). The
POST-run human gate will fail until `data/human_validation_sample_run2b.csv`
is labelled — expected, recorded, not a surprise. One execution at the
committed config; no re-filtering after labels.
