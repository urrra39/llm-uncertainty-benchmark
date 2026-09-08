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
(25–65% band, the band `configs/run3_gpu.yaml` ships), the run is a record of
that failure, not an invitation to a third pilot: the gate permits two
iterations and both are budgeted.

## Pilot contingency, pre-registered (added when run #3's slice moved to quantile 0.5)

Run #3's PopQA slice is harder than run #2b's (popularity quantile 0.9 → 0.5)
at the same time as the subject model is stronger (0.5B → 3B). Those are two
opposing forces on the base rate, and the config deliberately defers the
question to the pilot gate rather than guessing. Because both pilot
iterations could land outside the band, the failure path is fixed here, in
advance, so it cannot become post-hoc tuning:

- **Knob that moves: the popqa:triviaqa subset ratio only.** The question-set
  construction (relations, quantile, filters) is frozen by this document;
  changing it mid-pilot would alter what run #3 is a study of. The ratio is the
  one lever run #2's own pilot used, and the mix stays within the balanced
  design: never below 150 rows in either subset.
- **Direction:** measured against the error-rate band [0.25, 0.65]. If the
  pooled error sits above 0.65 (too hard), move 150 rows (25% of n) from the
  higher-error subset to the lower-error one. If it sits below 0.25 (too
  easy), move 150 rows the other way.
- **Magnitude per iteration:** exactly 150 rows per move (300/300 → 450/150
  worst case). No finer tuning within an iteration.
- **Maximum iterations before proceeding anyway:** two. If the second pilot
  still lands outside the band, the full run proceeds at the best mix with the
  gate failure recorded in its results file as the primary result — the base
  rate is then a documented finding, not a hidden tuning failure, and no
  signal is read as a ranking against a degenerate base rate (run #1's
  lesson). The two human gates and the class floor still apply unchanged.

## Judge cross-validation at n=600 (added with the config pre-flight)

`configs/run3_gpu.yaml` ships `judges.cross_validation_n: 120`, inherited from
run #2 where 120 ≥ the 66 judged rows, so the second judge covered every row
the primary judge saw (κ over the full judged set, D11). At n=600 that value
would cap the second judge at 120 rows — a fifth of the run — and the
judge-agreement κ would rest on that fifth. This was raised to **600** so a
credentialed run #3 keeps run #2's property: every primary-judged row is
second-judged, and the κ denominator equals the judged-row count by
construction (D11's assertion). The extra cost is bounded: the secondary judge
runs only on rows the primary judge was asked about, so `cross_validation_n`
is a ceiling, not a budget. The heuristic (no-key) path is unaffected.

## Power

Hanley–McNeil analytic interval at 150 positives / 150 negatives, AUROC 0.60:
[0.536, 0.664], half-width 0.064 — computed from the project's own
`analytic_auroc_ci`, not guessed. A true AUROC of 0.60 separates from 0.50 on
a balanced 300-row subset; run #2's 36/54 PopQA split could not (half-width
0.120). This is an approximation (it assumes no ties); the run reports
bootstrap intervals and this number only sizes the study.

## Per-dataset class counts: the required n (added after run #2b)

Run #2b measured PopQA 38% incorrect and TriviaQA 80% incorrect. The
`per_dataset_class_counts` gate demands ≥30 per class *within* each dataset,
so with those base rates the next run needs PopQA n ≥ 30/0.38 = 79 and
TriviaQA n ≥ 30/0.20 = 150 — unequal dataset sizes, or stratified sampling to
a balanced class mix within equal sizes. Equal-n splits at unequal base rates
relocate run #1's failure mode. Conditionally: IF run #3's base rates
resemble run #2b's, 300/300 gives expected minority rows of 114 and 60, both
clear of the floor — but base rates move with model scale and the new
relations, so the pilot gate (25–65%), not this paragraph, is the enforcement.

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

## E4 outcome lines, pre-written (D4)

When human labels land, the fuzzy-rule accuracy report decides which
paragraph below replaces the E4 bullet. Exactly one will be pasted; the other
two stay here as the record that the alternatives were live possibilities.

- **Rule accurate** (precision ≥ 0.90 with the lower interval bound above
  0.80): "Human labels confirm the fuzzy rule (precision X [l, h] on N
  fuzzy-decided rows); the E4 move stands as decontamination, with the
  judge-vs-heuristic caveat retained."
- **Rule unreliable** (precision below 0.80 or interval spanning chance):
  "Human labels reject the fuzzy rule (precision X [l, h]); E4 is withdrawn
  on the same footing as run #2, and the heuristic-labelled tables join it
  in the withdrawn document. The measured rule error, not a rescued ranking,
  is the finding."
- **Mixed** (rule accurate on one dataset / verdict class and not another):
  "Human labels split the rule's verdicts (numbers); E4 is reported
  per-subset with the failing subset withdrawn, and no pooled claim is made."

## Gates and stopping

Same five gates (random-baseline CI, ≥30 per class per subset where
applicable, abstention < 0.10, protocol ≥ 0.50, run coverage ≥ 0.80). The
POST-run human gate will fail until `data/human_validation_sample_run2b.csv`
is labelled — expected, recorded, not a surprise. One execution at the
committed config; no re-filtering after labels.

## Addendum (Round 11 / publication pass): the gate targets moved to the fixed set

This section pre-registered the run as executed. Round 11 re-labelled the run
under the fixed no-judge rule, and the publication pass made that corrected
set the published one (`results_run2b_fixedlabels.json`). A human gate on the
published set therefore reads the FIXED-label targets, not the pre-fix files
named above: `data/fuzzy_decided_rows_fixed.csv` for `human_label_coverage`
and `data/human_validation_sample_run2b_fixed.csv` for
`labeling_protocol_validated` (`configs/run2b_fixedlabels.yaml` resolves both).
The pre-fix files above remain committed as the Round-11 evidence; labelling
them measures the old rule, which is no longer the published set.
