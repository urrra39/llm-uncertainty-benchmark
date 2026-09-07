# Which cheap uncertainty signal best predicts an LLM's factual errors?

> **Run #2's ranking is withdrawn** to [docs/WITHDRAWN_RUN2.md](docs/WITHDRAWN_RUN2.md):
> up to 34 of its 120 labels (28%) could flip under the echo-contamination
> bound, past what any interval absorbs. Nothing withdrawn is quoted as a
> finding anywhere below. The primary result is run #2b's, shown as a
> measurement pending its human gates (see status block).

> Status, generated from `results_run2b.json` (`scripts/render_readme_header.py`):
> Primary run: run2b_clean (n=120, 71 incorrect / 49 correct) [results_run2b.json#run_name, views.primary.n, views.primary.n_incorrect, views.primary.n_correct].
> VALIDITY FAILED: per_dataset_class_counts, labeling_protocol_validated, human_label_coverage as recorded in results_run2b.json#validity_gates (t_random pooled, results_run2b.json#views.primary.signals.t_random: 0.523 [0.416, 0.628]).
> Label quality: 0/100 human-labelled (data/human_validation_sample_run2b.csv ROW:human_label) — lower bound on machine-label error 6/120 (5.0%) proven by code (data/label_error_audit.json); human labels are the only upper bound.

## Run #2b (primary, pending the human gate)

**Null result, stated first: at n=120 with stratified bootstrap intervals,
this benchmark cannot separate the top group.** 23 of 27 signals have
stratified intervals overlapping the leader's [0.658, 0.830] end to end —
18 of 22 distinct scored signals, plus the 5 rank-equivalent duplicates that
overlap by construction — the counts are computed from interval overlaps in
`results_run2b.json`, not hand-counted, and the table below marks the band. Same headline run #2 had,
now with the intervals to prove it rather than assert it.

<!-- TENSION:BEGIN -->
Why the null leads while rejections print below: the two tests answer different questions. Marginal-interval overlap asks whether two point estimates can be told apart on their own — the conservative read, and why the null leads. The paired bootstrap on AUROC differences asks whether one signal beats another on the same rows, exploiting their correlation; that is the test with the power to reject, and it is what rejects the trivial baselines. Here: 23 signals share the leader's band [0.658, 0.830], while `a_first_token_logprob`, `a_first_token_margin`, `a_mean_top5_entropy`, `c_verbal_confidence`, `t_random` are significantly worse.
<!-- TENSION:END -->

Run #2b re-ran run #2's exact configuration (Qwen2.5-0.5B-Instruct, same
decoding, prompts, sampling, NLI, n=120) with the decontaminated dataset:
no `capital of`, gold-in-question rows dropped, near-duplicates deduped, and
a balanced 60/60 split. Pre-registered before execution
(`docs/PREREGISTRATION.md`, "Run #2b" section). 27 of 27 registered signals
scored — the divergence is closed with real numbers.

Two honest caveats travel with every number below. First, **labels are
heuristic**: no judge credentials exist in an offline environment, so 47 rows
settled by exact match and 73 by the fuzzy containment rule, with no kappa —
and the rule's error is now measured only from below, not unbounded:
`scripts/audit_label_errors.py` proves 6 of 120 labels wrong (2 subject-echo
false positives, 4 verbatim-correct answers rejected on the length gap), with
per-row evidence in `data/label_error_audit.json` (a lower bound, because a
code sweep cannot see semantic errors a human would). Human labels are the
only upper bound (labelling block below).
Second, **three validity gates fail**: `per_dataset_class_counts` (both
columns below the ≥30 floor — see the banner on the table),
`labeling_protocol_validated` and `human_label_coverage` (both 0.0), so by
the project's own rule the ranking is not yet publishable. It is shown here
as a measurement with that status attached, not as a finding. What to label,
in what order, and which gate each file opens is generated below — read it,
not the file names around it.

<!-- LABELLING:BEGIN -->
To open the gates, label 59 rows, shared-first: `data/fuzzy_decided_rows.csv` (59 of 73 for `human_label_coverage`), then `data/human_validation_sample_run2b.csv` to 53 of 100 for `labeling_protocol_validated`. Wall clock: 59 rows x per-row rate (unmeasured; roughly 19-39 min at 20-40 s/row). Gate names: `human_label_coverage`, `labeling_protocol_validated`.
<!-- LABELLING:END -->

The three classical gates pass: `t_random` scores 0.445 [0.296, 0.598] on
PopQA and 0.571 [0.384, 0.752] on TriviaQA (both contain 0.50), classes are
71/49 overall, abstentions 0. Base rates: PopQA 23/37 incorrect (38%),
TriviaQA 48/12 (80%).

Full gate roster, generated from `results_run2b.json#validity_gates` (three
pass, three fail — the ranking is unpublished until all pass):

| gate | status | observed |
|---|---|---|
| random_baseline_ci_contains_chance | PASS | AUROC 0.523 [0.416, 0.628] |
| minimum_rows_per_class | PASS | 71 incorrect, 49 correct |
| abstention_rate_below_ceiling | PASS | 0/120 = 0.000 |
| per_dataset_class_counts | FAIL | popqa 23/37; triviaqa 48/12 |
| labeling_protocol_validated | FAIL | coverage 0.000 |
| human_label_coverage | FAIL | coverage 0.000 |

### Run #2b per-dataset AUROC (60/60, real bootstrap intervals)

> Column power: PopQA's minority class is 23, TriviaQA's is 12 — both below
> the project's own ≥30 floor (`per_dataset_class_counts` fails; run #2b's
> committed gates record it). The TriviaQA column must not be read as a
> ranking. The next run must target ≥30 per class *within* each dataset.

Selection rule, stated so curation cannot hide in it: all 22 distinct scored
signals (5 rank-equivalent duplicates in the appendix), sorted by stratified
AUROC descending, PopQA AUROC breaking ties. Rendered order is asserted in
tests against `results_run2b.json`. Stratified means carry paired bootstrap
intervals from one shared draw sequence across signals — and the sort order
does not survive them: the leader's [0.658, 0.830] overlaps the next seven
intervals end to end, so the ranking above is an ordering of point estimates,
not an ordering the data supports.

| signal | PopQA (23/37) | TriviaQA (48/12) | stratified |
|---|---|---|---|
| `b_disagreement_rate` | 0.768 [0.646, 0.882] | 0.727 [0.592, 0.847] | 0.747 [0.658, 0.830] |
| `a_total_logprob` | 0.825 [0.711, 0.922] | 0.656 [0.514, 0.790] | 0.741 [0.652, 0.823] |
| `a_length_normalized_logprob` | 0.811 [0.690, 0.914] | 0.656 [0.516, 0.790] | 0.734 [0.643, 0.818] |
| `b_distinct_count` | 0.742 [0.619, 0.857] | 0.712 [0.567, 0.842] | 0.727 [0.635, 0.814] |
| `b_mean_pairwise_f1` | 0.751 [0.627, 0.868] | 0.666 [0.495, 0.821] | 0.708 [0.604, 0.805] |
| `b_disagreement_rate_samples_only` | 0.729 [0.604, 0.850] | 0.673 [0.504, 0.826] | 0.701 [0.598, 0.798] |
| `b_distinct_count_samples_only` | 0.724 [0.600, 0.843] | 0.674 [0.505, 0.826] | 0.699 [0.597, 0.795] |
| `a_min_logprob` | 0.765 [0.635, 0.881] | 0.608 [0.446, 0.766] | 0.686 [0.584, 0.784] |
| `b_mean_pairwise_f1_samples_only` | 0.726 [0.600, 0.847] | 0.646 [0.465, 0.809] | 0.686 [0.578, 0.788] |
| `b_semantic_entropy` | 0.656 [0.525, 0.783] | 0.696 [0.539, 0.838] | 0.676 [0.577, 0.771] |
| `a_max_top5_entropy` | 0.760 [0.631, 0.877] | 0.566 [0.399, 0.726] | 0.663 [0.560, 0.762] |
| `c_p_true_plain` | 0.795 [0.660, 0.912] | 0.528 [0.311, 0.733] | 0.661 [0.537, 0.785] |
| `b_semantic_entropy_samples_only` | 0.638 [0.508, 0.766] | 0.664 [0.481, 0.828] | 0.651 [0.542, 0.757] |
| `c_p_true_with_samples` | 0.736 [0.593, 0.861] | 0.564 [0.391, 0.732] | 0.650 [0.539, 0.755] |
| `t_answer_length` | 0.603 [0.498, 0.709] | 0.650 [0.513, 0.774] | 0.626 [0.539, 0.710] |
| `a_mean_logprob` | 0.740 [0.609, 0.861] | 0.490 [0.319, 0.661] | 0.615 [0.507, 0.721] |
| `a_mean_top5_entropy` | 0.722 [0.586, 0.845] | 0.436 [0.283, 0.594] | 0.579 [0.477, 0.680] |
| `a_first_token_logprob` | 0.595 [0.441, 0.743] | 0.517 [0.340, 0.694] | 0.556 [0.438, 0.671] |
| ··· below this line: stratified interval entirely below the leader's [0.658, 0.830] ··· | | | |
| `t_question_length` | 0.499 [0.392, 0.611] | 0.553 [0.371, 0.730] | 0.526 [0.419, 0.629] |
| `c_verbal_confidence` | 0.504 [0.368, 0.639] | 0.547 [0.383, 0.707] | 0.525 [0.419, 0.634] |
| `t_random` | 0.445 [0.296, 0.598] | 0.571 [0.384, 0.752] | 0.508 [0.388, 0.629] |
| `a_first_token_margin` | 0.592 [0.434, 0.744] | 0.382 [0.222, 0.552] | 0.487 [0.373, 0.599] |

Five rank-equivalent duplicates (`b_distinct_fraction`,
`b_semantic_entropy_normalized`, `a_perplexity`,
`b_distinct_fraction_samples_only`,
`b_semantic_entropy_normalized_samples_only`) are appendix material, as
established: identical numbers, no information, no Holm penalty.

### What decontamination did (the E4 prediction, tested)

- **`a_mean_logprob` on PopQA moved 0.514 (withdrawn run-#2 baseline,
  directional only) → 0.740, clearing chance
  [0.609, 0.861] — but the move is NOT established as decontamination.**
  The permissive containment branch is inert on these 120 rows (fired 0
  times), so generosity in that branch cannot explain the move — and that is
  a NULL-POWER contrast: fuzzy and strict are the same label set here, so
  0.000 with interval [0.0, 0.0] is true by construction and the interval is
  degenerate (marked as such in `data/label_rule_sensitivity.json`, which
  refuses to report a rule-out from it). Containment-vs-TRUTH error is now
  measured from below rather than unbounded: `scripts/audit_label_errors.py`
  proves 6 of 120 labels wrong — 2 subject-echo false positives and 4
  verbatim-correct answers rejected on the length gap, all six inside the 73
  fuzzy-decided rows — and only human labels bound that error from above
  (`data/label_error_audit.json`). "Confirmed strongly" is
  withdrawn; the supported reading is "moved up under heuristic labels,
  mechanism unattributed". Under stratified intervals the move does not
  separate it from the group either: `a_mean_logprob` sits at 0.615
  [0.507, 0.721], inside the 23-signal band — so E4 is not the surviving
  finding; the null result is.
- **`c_verbal_confidence` on PopQA: 0.395 (withdrawn run-#2 baseline,
  directional only) → 0.504.** Predicted up; confirmed
  in direction only — it sits at chance, not above it. The hypothesis is
  right about the mechanism and overclaims nothing about this signal.
- **`t_question_length` stripped of provenance:** pooled 0.684 in run #2,
  0.499 on PopQA and 0.526 stratified here. It never measured uncertainty.
  Note run #2b's own pooled value is 0.719 — the pooled-vs-stratified gap
  WIDENED on clean data (0.719 vs 0.526), because the 60/60 split kept the
  base-rate asymmetry (38% vs 80% incorrect) that pays provenance detectors.
- **`t_answer_length` at 0.603 on PopQA beats real signals.** A token count
  of the answer outranks `a_first_token_logprob` (0.595),
  `c_verbal_confidence` (0.504) and `a_first_token_margin` (0.592) on the
  primary subset. This is the finding the project exists to surface: when a
  length baseline beats a logprob, the logprob is mostly rediscovering length
  at higher cost — the length-confound check the per-signal table already
  carries, now with a name on it.
- **`t_random` is now significantly worse than the leader** (−0.276,
  p_holm = 0.0076 over 21 distinct comparisons, 5 worse-significant total).
  Run #2's non-rejection was power (n=120, Holm over 20), not a defective
  test — the B1 calibration and this movement agree.
- **Pooled leader by point estimate is family A** (`a_total_logprob` 0.799
  [0.717, 0.872], overlapping the runner-up's [0.709, 0.865]), with no winner
  (gap 0.008, CIs overlap). The samples-only variants trail their
  greedy-included twins by 0.02–0.04, so the temperature-mixing bias is small
  on this run but measured rather than assumed.
- **N-ablation does not saturate at N=3 here:** 0.641 / 0.700 / 0.742 / 0.765
  at N=1/2/3/5, with N=1 significantly below N=5 (−0.124, p_holm = 0.016)
  and N=3 vs N=5 indistinguishable (−0.023, p = 0.698). Run #2's "use N=3"
  survives as cost advice, not as an optimum.
- **Clustering audit: measured, and small.** 4 disagreements in 120 audited
  rows (rate 0.033, Wilson 95% [0.013, 0.083]) — greedy single-pass and
  transitive-closure partitions agree on 97% of answer sets, so the published
  family-B numbers stand under either clusterer. Full audit
  (`force_full_audit`), not the default 20% sample. Family-B token multiplier
  measured at 6.01× for 6.0× calls — the call-count price was honest.
- **Run #2b's PopQA column is effectively one template, measured.** 59 of 60
  PopQA rows ask "What is the capital of X?" — the 90th-percentile popularity slice
  of the four configured relations is ~92% capital (184 of 199 unique
  questions), so the relation filter named four relations and the draw
  delivered one (plus a single `sport` row). Within-PopQA question length
  therefore varies only with the country name, not with question structure,
  which is why `t_question_length` sits at chance (0.499) there. Run #3's slice
  (quantile 0.5, seven relations) was checked against the source and is
  genuinely diverse.

Figures (all drawn from `results_run2b.json` alone):
`figures/run2b/auroc.png`, `figures/run2b/risk_coverage.png`,
`figures/run2b/reliability.png`, `figures/run2b/correlation.png`,
`figures/run2b/n_ablation.png`, `figures/run2b/cost_vs_auroc.png`.

## History of withdrawn runs

- **Run #1** (tag `run1-n100`): 7 correct of 75 answered; `t_random` at
  0.746 proved the estimator was generating the ordering. Full account in
  [docs/WITHDRAWN_RUN2.md](docs/WITHDRAWN_RUN2.md#run-1-and-why-it-was-discarded).
- **Run #2** (`configs/run2.yaml`): withdrawn over the echo-contamination
  bound (up to 34/120 labels flippable). All tables, findings and narrative
  preserved verbatim in [docs/WITHDRAWN_RUN2.md](docs/WITHDRAWN_RUN2.md).

## Reproduction (run #2b)

```bash
uv sync --extra local
uv run unc-bench build-dataset  --config configs/run2b_clean.yaml
uv run unc-bench generate       --config configs/run2b_clean.yaml
uv run unc-bench score-signals  --config configs/run2b_clean.yaml --family b
uv run unc-bench score-signals  --config configs/run2b_clean.yaml --family actc
uv run unc-bench ablation       --config configs/run2b_clean.yaml
uv run unc-bench label          --config configs/run2b_clean.yaml  # heuristic: no judge key here
uv run unc-bench analyze        --config configs/run2b_clean.yaml
uv run unc-bench figures        --config configs/run2b_clean.yaml
```

Family B runs as its own pass. Labels fall back to exact match plus fuzzy
containment without judge credentials (recorded, not hidden). Planned next:
run #3 at n=600 on a GPU (`configs/run3_gpu.yaml`, pre-registered in
[docs/PREREGISTRATION.md](docs/PREREGISTRATION.md)).


## License

MIT. See [LICENSE](LICENSE).
