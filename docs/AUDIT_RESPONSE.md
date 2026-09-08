# Audit response: item-by-item disposition

Each item of the external audit maps to the commit that addressed it or to an
explicit reasoned refusal. Silence is not a deliverable; disagreement with
evidence is.

> Appendix, not the record. This file preserves, item by item, how each
> external-audit item and round was disposed of. Every decision or measurement
> here that carries forward is logged in its decision form in
> [DECISIONS.md](DECISIONS.md) (the R-numbers) and tracked in
> [OPEN_DEFECTS.md](OPEN_DEFECTS.md); this file keeps the response context
> around them. For the current state, read the README and DECISIONS.md, not
> this appendix.

## Part A

- **A1 (echo pathology).** Done. Evidence first:
  `scripts/diagnose_echo_contamination.py` +
  `data/echo_contamination_report.json` (21 inverse rows, 20 echoes, 14 decided
  by subject-in-alias-list). `capital of` removed from `configs/run3_gpu.yaml`
  (`run2.yaml` untouched); `INVERSE_RELATIONS` guard with
  `allow_inverse_relations` hatch in `PopQABuilder`; `gold_in_question` check
  in both builders behind `drop_gold_in_question` (run #3 on); counts flow via
  `dataset_meta.json` into `results.json` `dataset.gold_leakage`. Correction
  to the audit: the echo holds on 20 of 21 visible instances, not 21 of 21.
- **A2 (near-duplicates).** Done. `questions_to_frame` raises on duplicate
  normalized text; `build` dedups with alias-list merging (logged, counted);
  run #3 comments re-measured (PopQA 318 unique, TriviaQA 3746);
  cluster-bootstrap option on all three estimators, off by default, on for run
  #3; singleton clusters reproduce row draws exactly at a fixed seed (tested).
- **A3 (human labels).** Done to the extent a model can: `docs/HUMAN_LABELING.md`
  protocol with an explicit echo rule; `human-agreement` extended (kappa CI,
  minority counts, per-source breakdown retained, oracle attenuation ceiling);
  `results.json` `labels.label_quality` (null + reason until filled); fourth
  gate `human_label_coverage` failing honestly at 0.0. The column itself is
  still empty — filling it is human work by definition.

## Part B

- **B1 (bootstrap audit).** Measured, then refused with evidence: 200 null
  trials hold type-I near 0.05; planted 0.196 gap has power
  (`tests/test_bootstrap_calibration.py`). No null-centred estimator added;
  published p-values stay traceable to one procedure.
- **B2 (per-dataset intervals).** Done as far as honesty allows: trackability
  test extended to every run artifact including the three new JSONs;
  Hanley–McNeil path already labels its method and now has an audit check.
  Real intervals need run #3's rows, which do not exist yet.
- **B3 (N-ablation test).** Done in infrastructure: `level_differences` with
  paired bootstrap + Holm across (signal, pair); README downgraded to "no
  measurable gain in this single draw" with the method named. Run #2's numbers
  unchanged (its artifact predates the block).
- **B4 (duplicates).** Done: `SignalSpec.rank_equivalent_to` declared for all
  three pairs, asserted empirically at analysis time (raises on config drift),
  Holm corrected over the 17 distinct with the 20-comparison table retained;
  README ranking shows 18 rows, appendix the 3 duplicates with identical numbers.

## Part C

- **C1 (reopen D27).** Done: width reverted to 1 in run #3 config and in the
  `ModelSpec` comment; three findings recorded; invariance claim scoped to
  measured hardware; `nondeterminism` extended into a batch-invariance harness
  (widths 1/2/4/8 × greedy text/logprobs/sampled texts, device+dtype recorded,
  pass rule named). Notebook estimate stands (it already assumes width 1).
- **C2 (per-sample RNG).** Done: `seed_for_question` (SHA-256 over base, qid,
  index); `generate_batch(seeds=...)` sub-splits buckets by seed; sampling loop
  uses per-question seeds; cross-width identity test pins it.
- **C3 (recovered κ + 66/60).** Done: recovered file recomputed from committed
  rows (κ=1.0, minority 6, trustworthy:false, degenerate CI stated);
  D26 correction marks 60-vs-66 a session boundary (run #1 vs run #2);
  audit_docs pins the recovered status and the reconciliation note.

## Part D

- **D1 (token costs).** Done additively: `cost_table` takes optional token
  means from the generations artifact (family C null with reason); figure
  prefers tokens with a call-multiple fallback so run #2's committed PNGs
  stand. Full token pricing needs verification token counts no artifact
  stores — recorded in the cost method string, not invented.
- **D2 (samples-only signals).** Done: six `_samples_only` signals scored
  beside the greedy-included six (plurality rule documented); orientation
  table extended; the samples-only entropy is named as the
  publication-comparable variant.
- **D3 (exhaustive audit).** Done: transitive-closure clusterer with tests
  (including the chain greedy splits); per-row audit in `run_b` stored in
  `family_b_meta.json` and folded into `results.json` as `family_b_clustering`.
- **D4 (dependency cleanup).** Done: sklearn/scipy already removed; tree-wide
  grep confirms zero runtime imports; CI step `check_imports.py` imports all
  36 modules on dev deps only.
- **D5 (defect tracker).** Done: `OPEN_DEFECTS` structured list in
  `scripts/audit_docs.py`, `docs/OPEN_DEFECTS.md` generated from it, audit
  fails on drift.

## Part E

- **E1 (pre-registration).** Done: `docs/PREREGISTRATION.md` committed before
  any run #3 number exists, with measured (not guessed) power arithmetic.
- **E2/E3 (run #3, second model).** Refused as unactionable here: no GPU, no
  judge credentials in this environment. The config, notebook and gates are
  ready; the runs are human/operator work.
- **E4 (trivial baselines primary).** Done in framing: stratified table shows
  `t_question_length` at 0.524 once provenance is removed; per-dataset table
  stays primary; pooled table is explicitly not to be quoted.
- **E5 (repo description).** Cannot be done from the working tree: the
  description is a GitHub setting. Suggested replacement: "n=120 null result:
  no uncertainty signal established above chance on PopQA; stratified table
  inside." Owner action required.

---

# Round 4: closing the audit gap + run #2b (executed)

## Withdrawal bound

Worst case from committed artifacts: 14 alias-decided echo rows observed in
100 plus all 20 unobserved rows, so up to **34 of 120 labels (28%)** could
flip. That bound exceeds every interval in the run and withdraws the ranking
(see README "Run #2 and why its ranking is withdrawn"). Item-by-item
dispositions for round 4 follow as each part lands.

## Dispositions

- **A1–A3.** Done (see round-3 entries for the machinery). New: README scope
  captions on every ranking table; `signal_coverage` block with a registry
  check that raises on unregistered columns (fixed one live bug: meta columns
  tripped the first version); sensitivity bound computed, ranking withdrawn
  in run-#1 style.
- **B1–B2.** Done: POST-run `human_label_coverage` (0.80) + PRE-run
  `labeling_protocol_validated` (0.50) as distinct gates; numbered ordering in
  the pre-registration; OPEN_DEFECTS regenerated; audit pins both names in
  all three documents.
- **C1–C3.** Done: PopQA pool widened to 2074 unique (6.91x) at quantile 0.5
  with the tail tradeoff recorded; relations selected from
  `data/gold_quality_report.json` (`place of birth` cut, `genre` in);
  `sampling_margin` in results + build warning below 2x. C3 refused with
  evidence (git log -S shows 0.25 since introduction; equality pinned).
- **D1–D2.** Done: sampled audit (seeded 20%, full-audit flag, Wilson
  interval, primary clusterer configurable defaulting to exhaustive);
  token prices from the generations artifact with family C null + reason;
  figure prefers tokens with call fallback; NLI cost re-measurement lands
  with run #2b's real timings (family B 0.21 s/item here).
- **E1–E6.** Done: `configs/run2b_clean.yaml` (60/60, verified distinct from
  run #2 only where decontamination requires + new-code defaults);
  pre-registered with measured power (half-width 0.145 — validity run, not a
  ranking study) and the falsifiable E4 prediction; executed on CPU here
  (6.1 s/question); `results_run2b.json` + 6 figures + tracked per-row
  artifacts committed; `data/human_validation_sample_run2b.csv` shipped
  49/51 with `human_label` empty.
- **E4 verdict.** `a_mean_logprob` PopQA 0.514 → 0.740 clearing chance:
  confirmed. `c_verbal_confidence` 0.395 → 0.504: direction confirmed,
  magnitude not — still chance. `t_question_length` 0.684 pooled → 0.499
  PopQA: provenance confirmed. `t_random` significantly worse than the leader
  on clean data: the estimator stands vindicated. Reported as directional
  confirmation because the labeler changed (judged → heuristic) alongside
  decontamination; a clean causal attribution needs judged labels.
- **F1–F3.** Done with one marked-unresolved: 71 (run #1 tag) and 66 (run #2
  file + D11 assertion) demonstrated; transient 60 in no committed config,
  marked unresolved in the D26 note. Audit cross-checks the withdrawal bound,
  gate names, recovered-κ status and reconciliation note; audit_docs runs in
  CI; README header generated from the primary results file (now run #2b's,
  honestly showing VALIDITY FAILED on the two human gates).

---

## Settled refusals

Each stated once, with the round that settled it and its evidence. Later
rounds point here instead of restating. Rounds are numbered by audit round;
"settled in round N" means first argued with evidence there.

- **E4 numbers stay; attribution stays withdrawn (settled round 6).** The
  0.514 → 0.740 move is measured directional evidence under stated labelers;
  deleting it would delete evidence. The causal claim was withdrawn, replaced
  with the null-power admission.
- **No length-bias defect (settled round 5).** Generous branch fired zero
  times in 120 rows; differential bias exactly 0.000. A defect needs a
  measurement.
- **No partial clustering raise (settled round 7).** Full-audit rerun
  executed instead: 4/120, Wilson [0.013, 0.083]. The earlier downgrade
  stands as correct on its evidence (R18).
- **No "27 vs 30" reconciliation (settled round 5).** Registry recounted in
  code: 27 signals. No committed file, document or table counts 30 signals.

---

# Round 5: documentation integrity + run #2b confound (executed)

## Refused with reason

- **P3.2 as written is stale in two of three clauses.** The 318-unique pool
  was superseded two rounds ago: run #3 now draws 300 from 2074 unique
  (6.91x, measured in-config). The pilot-gate 0.35 was refused last round
  with git evidence and is refused again here rather than re-litigated. What
  was live — the alias diagnostic for the new relations — is done
  (`data/gold_quality_report.json` over all 16 relations; `place of birth`
  cut at granularity-span 0.620, `genre` in at single-alias 0.074).
- **P3.5's "27 vs 30 registry entries" has no referent.** The registry holds
  27 signals (21 + 6 samples-only), verified in code; no committed file,
  document or table ever counts 30 signals. The 30s in the repo are the
  per-class floor and run #2's TriviaQA n. There is nothing to reconcile.
- **P1.2's requested defect was not filed, with evidence.** The generous
  containment branch fired zero times in 120 rows; fuzzy and strict label
  sets are identical, so the differential length bias is exactly 0.000.
- **B1's null-centred estimator (round 4) stays refused.** Calibration holds
  type-I near nominal; run #2b additionally shows the same test rejecting
  `t_random` at p=0.0076 on clean data.

## Done

- **P0.1–P0.3.** WITHDRAWN_RUN2.md split with status markers; `results.json`
  renamed (config default is now `results_default.json`); header carries view
  names + source file with provenance per number; audit fails on unregistered
  README signals, README AUROCs absent from the primary file (pinned run #2b
  table), defects missing from OPEN_DEFECTS, and withdrawn numbers outside
  history. Failing output was produced first (empty registry, stale gate
  string, stray run #2 intro block) and each fixed.
- **P1.1.** Decomposition recomputed from the same generations under three
  rules: rule effect 0.000 [0.0, 0.0]. E4 stands. A first version reported
  0.481 through a sign bug (correctness stored where incorrectness was
  read); caught because 1 − 0.740 == 0.260 exactly, and recorded here rather
  than hidden.
- **P1.3–P1.4.** Quantile regime recorded per run (run #2b kept 0.9; run #3
  moved with a note); `comparability_note` assertion live with tests for both
  configs. Per-dataset class gate live, failing both thin run #2b columns as
  designed; banner on the table; design arithmetic (79/150) in pre-registration.
- **P2.** `label-human` loop (no prefill, machine hidden until commit, atomic
  resume, timing log, ambiguous logged separately) with tests;
  `--require-judges` aborts instead of silent fallback (unrun: no key);
  runbook rewrite; no-inter-labeler LIMITATIONS entry. Payoff is automatic on
  analyze re-run (label_quality + gates already wired).
- **P3.1.** Downgraded, not raised: ≥23 audited rows needed at the observed
  rate, 16 drawn — "unmeasured at useful precision", with the arithmetic.
- **P3.3.** Verified: width pinned at 1 with CPU/bfloat16 scoping in config,
  harness recording device+dtype, notebook estimate assuming no speedup.
  No equivalence claim exceeds its hardware.
- **P3.4.** Exact string for GitHub Settings → General → About (owner must
  paste; cannot be set from the tree): "Run #2b (n=120, CPU): no uncertainty
  signal established above chance on 60-row subsets; decontaminated validity
  run, heuristic labels, humans pending. Withdrawn run #2 inside."

## Reconciliation table (P3.5)

| Count | Authoritative source | Status |
|---|---|---|
| 66 judged rows, run #2 | `results_run2_withdrawn.json` labels.kappa.n + D11 assertion | demonstrated |
| 71 judged rows, run #1 | `run1-n100` tag results.json | demonstrated |
| 60-row transient request | no committed config | UNRESOLVED, marked in D26 note |
| 20 echo / 21 inverse rows | `data/echo_contamination_report.json` | demonstrated (20 of 21; exception named) |
| 27 registry signals | `SignalSpec` registry, counted in code | demonstrated; "30" has no referent |
| 34/120 withdrawal bound | echo report (14) + 20 unobserved, worst-cased | derived, pinned by audit |

## What is still unmeasured (flatly)

- Human labels: 0/100 on both samples. Everything downstream of correctness
  (attenuation bound, publishable ranking) waits on a human with a terminal.
- Judge labels on run #2b rows: no key, no kappa, no second opinion of any kind.
- Generation bit-reproducibility and batch invariance on the T4 in fp16.
- Whether the widened run #3 slice lands in the pilot band at 3B scale.
- Any per-dataset bootstrap interval for run #2 (artifacts lost permanently).

---

# Round 6: documentation integrity + run #2b confound (executed)

## Refused with reason

- **Withdrawing the E4 numbers entirely.** The move 0.514 → 0.740 stands as a
  measured directional delta under stated labelers; what was withdrawn is the
  causal attribution ("confirmed strongly"), replaced with the null-power
  admission and the unboundedness statement. Deleting the numbers would be
  deleting evidence (invariant 3).
- **Filing a length-bias defect.** The generous branch fired zero times, so
  fuzzy and strict label sets are identical and the differential bias is
  exactly 0.000. A defect describing a mechanism with zero firings would be a
  claim without a measurement.
- **Re-running run #2b's family B at full audit for a tighter interval.**
  The claim was downgraded instead (≥23 rows needed, 16 drawn): re-running
  model stages to narrow one sentence risks artifact churn for no finding.
- **P3.5's "27 vs 30 registry entries" (round 5, restated here).** Recounted
  in code: 27. No "30 signals" exists anywhere committed. Nothing to reconcile.

## Done

- **P0.1–P0.4.** E4 rewritten to the supported reading (branch inert, contrast
  null-power with the interval marked degenerate, 73/120 from the rule, humans
  the only bound). Power block asserted in code with detectable_effect false.
  `data/fuzzy_decided_rows.csv` ships the 73-row population, human_label
  empty, wired as the default labelling target. Withdrawn baselines marked
  inline (option (a), recorded in R14); the auditor now enforces the marking
  and exempts 0.514-class collisions with the stated reason.
- **P1.1–P1.4.** Primary table holds all 22 distinct signals under the stated
  rule, asserted in `tests/test_readme_numbers.py` — which caught two live
  hand-arithmetic errors on first run (c_p_true_plain stratified 0.662 →
  0.661; c_verbal_confidence/t_question_length order). t_answer_length has
  its bullet; t_question_length quotes run #2b pooled 0.719 with the widened
  gap; every table cell and headline number is pinned to results_run2b.json.
- **P2.1–P2.3.** Banner rewritten without dangling references (stale-scope
  phrasing fails the audit); header numbers carry view names and JSON
  pointers; full six-gate roster generated from the file.
- **P3.1–P3.3.** `--target` defaults to the fuzzy file; runbook rewritten
  around it with timing and the dropped-not-coerced fate of ambiguous rows;
  tests prove no write without an interactive verdict, correct resume, blank
  ambiguous storage, and an untouched committed tree. `--require-judges`
  aborts instead of silent fallback (unrun: no key here).
- **P4.1–P4.5.** `rule_accuracy` (precision/recall + paired bootstrap,
  hand-tested); `--human-csv` reference arm scoring every signal under human
  labels (verified on synthetic tmp labels with backup/restore; committed
  artifacts untouched); attenuation storage already wired into label_quality
  and gates; README/gates flip automatically on analyze re-run. Nothing
  executes until a human labels — by design, not by omission.
- **P5.1.** Downgraded with arithmetic (≥23 needed, 16 drawn), not split.
- **P5.2 run #3 pre-flight, verified item by item:** margin 6.91x ≥ 3x
  (measured in-config); alias diagnostic run against all 16 relations with
  `place of birth` cut and `genre` in on the numbers; per-class design
  arithmetic (79/150) in pre-registration; pilot band 0.25 confirmed
  identical across all four configs (the alleged 0.35 change refused twice
  with git evidence); D27 open, width pinned at 1 with CPU/bfloat16 scoping.
- **P5.3.** Exact string for GitHub Settings → General → Description (owner
  must paste; not settable from the tree): "Run #2b (n=120, CPU): no
  uncertainty signal established above chance on 60-row subsets;
  decontaminated validity run, heuristic labels, humans pending. Withdrawn
  run #2 inside."

## What is still unmeasured (flatly)

- Human labels on either sample: 0/100 and 0/73. Every downstream payoff
  (kappa, rule accuracy, attenuation, publishable ranking) waits on this.
- A second opinion of any kind on run #2b rows (no judges, no second human).
- T4/fp16 generation determinism and batch invariance.
- Run #3's base rate at 3B scale with four ungenerated-against relations.
- Run #2 per-dataset bootstrap intervals (artifacts lost permanently).
- Whether any signal beats chance on 60-row subsets: run #2b's intervals all
  admit it, which is a statement about n, not about the signals.

## Signal coverage (generated, Part B2)

<!-- SIGNAL_TABLE:BEGIN -->
| signal | family | rank-equivalent to | run #2 | run #2b |
|---|---|---|---|---|
| a_first_token_logprob | A | — | scored | scored |
| a_first_token_margin | A | — | scored | scored |
| a_length_normalized_logprob | A | — | scored | scored |
| a_max_top5_entropy | A | — | scored | scored |
| a_mean_logprob | A | — | scored | scored |
| a_mean_top5_entropy | A | — | scored | scored |
| a_min_logprob | A | — | scored | scored |
| a_perplexity | A | a_mean_logprob | scored | scored |
| a_total_logprob | A | — | scored | scored |
| b_disagreement_rate | B | — | scored | scored |
| b_disagreement_rate_samples_only | B | — | — | scored |
| b_distinct_count | B | — | scored | scored |
| b_distinct_count_samples_only | B | — | — | scored |
| b_distinct_fraction | B | b_distinct_count | scored | scored |
| b_distinct_fraction_samples_only | B | b_distinct_count_samples_only | — | scored |
| b_mean_pairwise_f1 | B | — | scored | scored |
| b_mean_pairwise_f1_samples_only | B | — | — | scored |
| b_semantic_entropy | B | — | scored | scored |
| b_semantic_entropy_normalized | B | b_semantic_entropy | scored | scored |
| b_semantic_entropy_normalized_samples_only | B | b_semantic_entropy_samples_only | — | scored |
| b_semantic_entropy_samples_only | B | — | — | scored |
| c_p_true_plain | C | — | scored | scored |
| c_p_true_with_samples | C | — | scored | scored |
| c_verbal_confidence | C | — | scored | scored |
| t_answer_length | T | — | scored | scored |
| t_question_length | T | — | scored | scored |
| t_random | T | — | scored | scored |

Tallies: 27 registered; 22 distinct scored in run #2b; 5 duplicates scored alongside.
<!-- SIGNAL_TABLE:END -->

---

# Round 7: gate mapping, stratified intervals, human path (executed)

## Refused with reason

See [Settled refusals](#settled-refusals).

## Done (with evidence, not prose)

- **A1–A4.** Gate/label mapping remade so every gate reads run-owned files:
  protocol gate reads the run's own sample (0.50), coverage gate reads the
  run's fuzzy rows (0.80). `docs/LABEL_GATE_MAP.md` generated from
  `GATE_SOURCES` plus resolved config paths, pinned by test. Mapping test
  both directions. Minimum honest cost recomputed: 59 rows (~20 min at an
  explicitly unmeasured per-row rate; the runbook's old "measured 20s" never
  was measured and now says so).
- **B1.** Paired stratified bootstrap (one shared draw sequence, seed
  recorded) populates the stratified column; sort does not survive
  (leader [0.658, 0.830] overlaps seven intervals) and the table says so
  beside the ranking.
- **B2.** Generated signal table (27/22/5) embedded in this file, pinned.
- **B4 triage.** README's remaining uncomputeds: label quality + ranking
  status (needs human, Part D), clustering audit (computed this round).
- **D1.** Skip/quit/garbage-path tests added; spec verified item by item.
- **D2.** Rule accuracy renders automatically with labels (tested present
  and absent); attenuation/gates/header flip on analyze re-run (mechanism
  verified, nothing to trigger it yet).
- **D4.** Three outcome paragraphs pre-written in pre-registration.
- **C1–C5.** Verified against current files (margins, diagnostics, design
  arithmetic, D27 open at width 1, prereg complete with E4 prediction).
- **E1.** `docs/CEILING.md` states the cap first.
- **E4.** Description string (P5.3, unchanged and still unpasted by the
  owner): "Run #2b (n=120, CPU): no uncertainty signal established above
  chance on 60-row subsets; decontaminated validity run, heuristic labels,
  humans pending. Withdrawn run #2 inside."

## What is still unmeasured (flatly)

- 59+ human labels (fuzzy file first, shared rows next). Nothing downstream moves without these.
- Any second opinion on run #2b rows; T4/fp16 determinism and batch invariance.
- Run #3's base rate at 3B scale; run #2's per-dataset intervals (lost permanently).
- Whether the E4 move survives human labels (the decomposition hook is built and waiting).

---

# Round 8: null band first, shared-first labelling, measured audit (executed)

## Refused with reason

See [Settled refusals](#settled-refusals).

## Done (measured results, not prose)

- **A1/A2.** Null result leads: 23 of 27 stratified intervals overlap the
  leader's [0.658, 0.830], count computed from overlaps; table separator
  above the 4 below-band rows, position asserted in tests.
- **A5.** E4 under stratified intervals: a_mean_logprob at 0.615
  [0.507, 0.721], inside the band — the move is not the surviving finding.
- **A3.** Superlative audit live; caught one ("Pooled leader is family A",
  now carrying [0.717, 0.872] vs [0.709, 0.865]).
- **B1–B4.** Intersection measured at 53 (not assumed); minimum 59 rows, not
  109; `label-plan` prints rows, shared-first order and wall clock with the
  rate marked unmeasured.
- **D1.** Skip/quit/garbage/timing/prefill tests close every spec clause.
- **D2.** Rule accuracy renders automatically iff labels exist (tested both
  directions); attenuation/gates/header flip on analyze re-run (mechanism
  verified, nothing to trigger).
- **D4.** Three outcome paragraphs pre-written in pre-registration.
- **C1–C5.** Verified item by item against current files (margins,
  per-relation diagnostics, design arithmetic, D27 open at width 1,
  complete pre-registration with E4 prediction).
- **E1.** `docs/CEILING.md` states the cap first.

## What is still unmeasured (flatly)

- 59 human labels minimum (fuzzy file first). The single blocker for validity.
- Any second opinion on run #2b rows; T4/fp16 determinism and batch invariance.
- Run #3's base rate at 3B scale; run #2's per-dataset intervals (lost).
- Whether E4's move survives human labels (hook built, waiting).
- Measured per-row labelling time (the runbook's band is an explicit guess).

---

# Round 9: null first, shared-first plan, measured audit (executed)

## Refused with reason

See [Settled refusals](#settled-refusals).

## Done (measured results)

- **A1/A2.** Null leads with the generated count (23/27; 18/22 distinct,
  both denominators from one overlap computation, asserted in tests);
  generated tension paragraph placed at the contradiction; table separator
  position asserted.
- **A5.** E4 sits inside the band (0.615 [0.507, 0.721]); the move is not
  the surviving finding.
- **A3.** Superlative audit live; caught and fixed one live instance
  ("Pooled leader", now carrying both intervals).
- **A4.** R18 (round 8): the 16-row downgrade stands as correct on its evidence.
- **B1–B4.** Intersection measured at 53; minimum 59 rows; `label-plan`
  prints rows, shared-first order and wall clock with the rate marked
  unmeasured.
- **D1.** Skip/quit/garbage/timing/prefill tests close every spec clause.
- **D2.** Rule accuracy renders automatically iff labels exist (tested both
  directions); attenuation/gates/header flip on analyze re-run (mechanism
  verified, nothing to trigger).
- **D4.** Three outcome paragraphs pre-written in pre-registration.
- **C1–C5.** Verified item by item against current files (margins,
  per-relation diagnostics, design arithmetic, D27 open at width 1,
  complete pre-registration with E4 prediction).
- **E1.** CEILING states the cap and now the replication line.

## What is still unmeasured (flatly)

- 59 human labels minimum (fuzzy file first, shared rows next). The single
  blocker for validity; no agent may do this.
- Any second opinion on run #2b rows; T4/fp16 determinism and batch invariance.
- Run #3's base rate at 3B scale; run #2's per-dataset intervals (lost).
- Measured per-row labelling time (the runbook's band is an explicit guess).
- Whether E4's move survives human labels (hook built, waiting).

---

# Round 10: process repair closed, owner work queued (executed)

Disposition states: DONE, REFUSED-WITH-EVIDENCE, BLOCKED-ON-OWNER. No item
below lacks one.

- **A-iv, "Confirm the stale labelling scope is actually gone" (round 10).**
  DONE. Offending paragraph moved to the withdrawn document; stale phrases
  fail the audit; verified clean.
- **A-i, "Fix by quoting each audit item VERBATIM (first ~15 words)" (round 10).**
  DONE. Every bullet below quotes its item; a test fails any quoteless
  disposition in this section.
- **A-ii, "Add a third disposition state — DONE / REFUSED-WITH-EVIDENCE /
  BLOCKED-ON-OWNER — and re-triage" (round 10).** DONE, this section is the
  implementation; open-item triage follows it.
- **A-iii, "There are TWO identical Round 4 sections" (round 10).** DONE.
  Stub deleted; settled refusals stated once with settling rounds; rounds 7–9
  point at it; audit enforces unique headings and no restatement.
- **A-v, "instrument per-row seconds in label-human, owner labels THREE rows
  only, hardcode the measured median" (round 10).** BLOCKED-ON-OWNER.
  Instrumentation exists and is tested (timing log with per-row seconds);
  the three rows do not, and no agent may produce them.
- **A1, "Rewrite the run #2b lead so the FIRST claim a reader meets is the
  null result" (round 10).** DONE. Null leads with the generated 23-count.
- **A2, "Emit BOTH from a single computation with the denominator named in
  each" (round 10).** DONE. "23 of 27" and "18 of 22 distinct" derive from
  one overlap computation, asserted in tests.
- **A4, "Quote the interval-consistent range, not the point estimate alone"
  (round 10).** DONE. Clustering bullet quotes [0.013, 0.083]; sweep found
  no other point-as-fact instance.
- **B1–B4, "Compute the actual intersection ... Publish the number" (round 10).**
  DONE. Intersection measured at 53 (not assumed); minimum 59 rows;
  `label-plan` prints rows, shared-first order and wall clock.
- **D1, "Verify `label-human` against its spec ... Add the missing tests"
  (round 10).** DONE. Skip/quit/garbage/timing/prefill tests close the gaps.
- **D2, "On coverage > 0, with no further prompting, produce" (round 10).**
  DONE in code (rule accuracy renders iff labels exist, tested both ways;
  attenuation/gates/header flip on re-run); BLOCKED-ON-OWNER for execution.
- **D4, "Write the one-paragraph README status line for each possible outcome
  in advance" (round 10).** DONE. Three paragraphs pre-written in
  pre-registration; exactly one to be pasted.
- **C1, "Report each pre-flight item as DONE / REFUSED / BLOCKED with
  evidence" (round 10).** DONE: margin 6.91x measured in-config;
  per-relation diagnostics published with cuts on the numbers; design
  arithmetic in pre-registration; D27 open at width 1 with hardware scoping.
- **E1, "Add docs/CEILING.md" (round 10).** DONE (rounds 7–8); replication
  line verified present this round.
- **P3.5-style "27 vs 30" recounts (round 10).** REFUSED-WITH-EVIDENCE (see
  Settled refusals): registry recounted in code at 27; no referent for 30.

## Open-item triage (A-ii)

| item | state | evidence / unblock condition |
|---|---|---|
| 59 human labels (fuzzy first) | BLOCKED-ON-OWNER | `label-plan` output; no agent may do this |
| 3-row timing instrumentation | BLOCKED-ON-OWNER | loop records seconds; needs 3 labelled rows |
| T4/fp16 determinism + batch invariance | BLOCKED-ON-OWNER | needs GPU hardware |
| run #3 base rate at 3B scale | BLOCKED-ON-OWNER | needs run #3 execution |
| run #3 + second subject model | BLOCKED-ON-OWNER | GPU + operator time |
| run #2 per-dataset intervals | closed, permanent | artifacts lost; Hanley–McNeil stands |

## What is still unmeasured (flatly)

- Human labels on either sample: 0/100 and 0/73.
- Any second opinion on run #2b rows; T4/fp16 determinism and batch invariance.
- Run #3's base rate at 3B scale; run #2's per-dataset intervals (lost).
- Measured per-row labelling time (three rows would instrument it).
- Whether E4's move survives human labels (hook built, waiting).

# Round 11: labeler errors reproduced, fixed, and measured (executed)

Validity dropped to 5.5 this round because a hand-check of
`data/human_validation_sample_run2b.csv` found demonstrable label errors. Those
errors are reproduced in code (`scripts/audit_label_errors.py`, 5/5), the rule
that made them is replaced, run #2b is re-labelled under the fixed rule with
both label sets committed, and the labeler-induced AUROC movement is measured.
Dispositions below: DONE / REFUSED-WITH-EVIDENCE / BLOCKED-ON-OWNER.

- **PART 0.1, "Write scripts/audit_label_errors.py that reproduces all five as failing" (round 11).**
  DONE. Reproduces all five against the committed run artifacts and sweeps all
  120 rows. The two systematic patterns are the rule's own fingerprint: the
  fuzzy rule marked exactly TWO rows correct and both are subject-echo false
  positives (`popqa-5864218`, "Jamaica" inside "Kingston, Jamaica";
  `triviaqa-jp_1520`, "whale" inside "Unicorn Whale"); four verbatim-correct
  answers were rejected on the length gap (`popqa-6298839`, `triviaqa-qb_1435`,
  `triviaqa-dpql_376`, and `triviaqa-bb_3148`, which sits in the 20 rows
  outside the 100-row sample the auditor read). Lower bound on machine-label
  error 6/120 = 5.0%, per-row evidence in `data/label_error_audit.json`.
- **PART 0.2, "Publish that lower bound in README beside every ranking" (round 11).**
  DONE. The header status, the caveat that travels with the table, and the E4
  bullet now carry the measured floor; "unbounded" is gone; the header
  generator reads the audit JSON so the number cannot drift from its source.
- **PART A1, "Echo guard in labeling: a containment match must NOT count as correct" (round 11).**
  DONE, within the no-judge rule rewrite below. A bare echo of the question's
  own words is incorrect; both false positives are regression tests
  (`tests/test_labeling.py`, `tests/test_labeling_round11.py`). Recorded in
  DECISIONS (R19–R21) and OPEN_DEFECTS (RUN2B-LABELSET).
- **PART A2, "Replace the gap cap with answer-span extraction ... explicit UNRESOLVED verdict" (round 11).**
  DONE. Correct requires a FULL gold alias present verbatim with every other
  token traceable to the question (no length gap to exceed); rule-unresolved
  rows are AMBIGUOUS under source `heuristic_unresolved` and go to the human
  queue, never silently coerced to incorrect. A1 and A2 land as one change
  because the five-case regression suite cannot pass on the echo guard alone:
  the "whale" false positive closes only with the answer-shorter containment
  removal, which is A2's span extraction — stated in DECISIONS, not hidden.
  On the 120 rows the fixed rule moves exactly the six demonstrable errors and
  nothing else (pinned).
- **PART A3, "Re-label run #2b with the fixed rule. Commit BOTH label sets" (round 11).**
  DONE. `data/run2b/labels_fixed.parquet` committed beside the pre-fix
  `labels.parquet`; `results_run2b_fixedlabels.json` produced by the same
  `analyze` pipeline (mirror config/artifacts dir) so it is schema-identical;
  `data/labeler_variance_run2b.json` carries every per-signal AUROC under both
  label sets with a like-for-like paired-bootstrap delta on the shared 119
  rows, after validating the estimator reproduces every committed AUROC point
  and CI bound exactly. Old numbers are not overwritten — they are the object
  of measurement. Counts move 71/49 → 68/51 at n=119 (one row rule-ambiguous,
  excluded and counted).
- **PART A4, "If the fixed labels move the ranking, the ranking moves. If they move E4, E4 moves." (round 11).**
  The movement is measured and published in the README round-11 section: pooled
  leader `a_total_logprob` 0.799 falls out; the PopQA column leader becomes
  `c_p_true_plain` 0.828; the stratified column-top `b_disagreement_rate`
  0.747 drops to 0.691, and `b_mean_pairwise_f1` 0.705 [0.601, 0.802] leads the
  fixed-label table. E4's `a_mean_logprob` PopQA reading becomes 0.698
  [0.553, 0.833] (still above chance). The "paste exactly one pre-written E4
  outcome paragraph" clause is BLOCKED-ON-OWNER: the three pre-written
  paragraphs in PREREGISTRATION are conditioned on the HUMAN fuzzy-rule
  accuracy report (Part D), and pasting one before that report exists would be
  editing the record after the fact on the wrong trigger. All three
  alternatives remain live until D.
- **PART B1, "`a_total_logprob` ... Quantify the overlap: partial Spearman ... AUROC within length strata" (round 11).**
  DONE. `data/length_confound_audit.json` scores every signal on the same 119
  rows under both label sets: Spearman with label and with answer length,
  rank-residualized partial Spearman given length, and within at/below- vs
  above-median-length AUROC with bootstrap intervals. `a_total_logprob`:
  label 0.519, length 0.594, partial 0.401; its above-median stratum 0.855
  [0.731, 0.951] under the old labels collapses to 0.714 [0.536, 0.873] under
  the fixed labels, flat against its short-stratum 0.711. `t_answer_length` is
  length by definition (Spearman 1.0, partial 0.128); `c_p_true_plain` is
  length-independent (0.049).
- **PART B2, "Publish a 'length-mediated share' column or an explicit paragraph naming which top-table signals lose their edge" (round 11).**
  DONE as an explicit paragraph: the deltas whose CI excludes zero are exactly
  the length-correlated signals (`a_total_logprob` −0.066 [−0.130, −0.015];
  `a_length_normalized_logprob` −0.056 [−0.115, −0.011]; `t_answer_length`
  −0.070 [−0.137, −0.016]), and they collapse toward the family-B band under
  the fixed labels — the round's headline.
- **PART B3, "State the mechanism plainly in README" (round 11).** DONE. The
  sentence stands in the README round-11 section: a length-biased labeler plus
  a length-correlated signal manufactures AUROC that measures neither
  uncertainty nor correctness.
- **PART C2, "Add an assertion that alias merging never unions aliases across distinct Wikidata subject QIDs" (round 11).**
  DONE. The PopQA builder drops whole any duplicate-question group whose
  Wikidata subject QIDs differ and asserts none survive to the generic
  alias-merging dedup; counts recorded in `dataset_meta.json`. Run #2b's 120
  rows contain exactly one affected row (`popqa-1782552`, "What is the capital
  of Georgia?", gold spanning country-Georgia {Kutaisi, Tbilisi, ...} and
  US-state-Georgia {Atlanta, ...}) — a row that cannot be answered wrongly.
  Run #3's slice has 38 such groups (81 rows); the guard drops them at build.
- **PART C1, "SINGLE-TEMPLATE COLLAPSE ... Verify against the built dataset" (round 11).**
  DONE. Measured: run #2b's PopQA draw is 59/60 "What is the capital of X?",
  because the 90th-percentile slice of the four configured relations is ~92%
  capital (184 of 199 unique). The card is fixed (LIMITATIONS item 4, README,
  config comment); the config is not changed because run #2b is a completed
  run. Run #3's slice (quantile 0.5, seven relations) was checked against the
  source and is genuinely diverse (2191 unique pre-leakage).
- **PART C3, "Re-check the base rates and both class-count gates after C1+C2 and A3" (round 11).**
  DONE. After A3 (C1/C2 do not alter run #2b's historical row set): pooled
  68/51 at n=119 (0.571); PopQA 23/37 (0.383); TriviaQA 45/14 (0.763).
  Gates: random-baseline PASS 0.538 [0.433, 0.639]; minimum_rows_per_class
  PASS 68/51; abstention PASS 0/119; per_dataset_class_counts FAIL — minority
  23 and 14, both under the ≥30 floor (D5's arithmetic is 23 and 14 under the
  fixed labels, not 23 and 12); the two human gates FAIL at 0.0.
- **PART D (owner), "`unc-bench label-plan`, then `unc-bench label-human`" (round 11).**
  BLOCKED-ON-OWNER. No agent may fill a `human_label` cell. Prep delivered:
  fixed-label targets `data/fuzzy_decided_rows_fixed.csv` (73 rows, the one
  rule-ambiguous "whale" row shipped with a blank fuzzy_verdict) and
  `data/human_validation_sample_run2b_fixed.csv` (100 rows balanced on the
  fixed machine label), produced by `scripts/export_fixed_human_files.py`;
  `unc-bench label-plan` prints 59 rows shared-first; every row flagged by
  PART 0.1 and the rule-ambiguous row lies inside the 73 fuzzy-decided rows
  (D2 is satisfied by labelling that population).
- **PART E1, "Report each pre-flight item DONE / REFUSED / BLOCKED with evidence" (round 11).**
  DONE, table below.
- **PART E2/E3 (owner), "run #3 at n=600 ... second subject model from a DIFFERENT family" (round 11).**
  BLOCKED-ON-OWNER (GPU + operator time).
- **PART F1/F2/F3 (authorship and stop condition).** F1 honoured (single
  author; no CONTRIBUTORS, no co-author trailers, no PRs). F2: one README line
  added under Methods/Acknowledgements. F3: CEILING.md already states the last
  half point is external replication, not a task; the pass stops after this
  round's measurement.

## E1 pre-flight (re-verified after the dataset changes this round)

| item | state | evidence |
|---|---|---|
| PopQA margin ≥ 3x for 300 draws | DONE | 2074 unique after leakage = 6.91x; the C2 guard removes 38 duplicate-question groups (81 rows) → ~2036 unique = 6.79x; both measured from the source, config comments carry the numbers. The 0.9-quantile census worry (318 unique) is retired: run #3 uses quantile 0.5. |
| Per-relation alias-completeness diagnostics BEFORE generation | DONE | `data/gold_quality_report.json` is committed and predates run #3: religion single_alias_fraction 0.050, occupation 0.147, genre 0.074; `place of birth` (0.250) was cut from run #3 on granularity-span 0.620 — the report covers it regardless. |
| ≥30 minority rows WITHIN each dataset | DONE | From run #2b's measured base rates: PopQA 300 × 0.38 ≈ 114 and TriviaQA 300 × 0.20 ≈ 60 minority rows, both clear of the floor (preregistered); conditioned on base rates surviving at 3B scale, which the pilot gate (25–65%) enforces. |
| D27 open | DONE | OPEN_DEFECTS; `generation_batch_size: 1` pinned until batch invariance passes on T4/fp16. |
| generation_batch_size 1 | DONE | `configs/run3_gpu.yaml`. |
| Relation diversity actually present | DONE | Run #3 PopQA slice: 7 relations, 2191 unique pre-leakage (capital 518, genre 872, occupation 200, religion 199, country 234, sport 135, color 33) — the C1 single-template trap does not recur. |

## What is still unmeasured (flatly)

- Human labels on any target: 0/100 and 0/73 on the pre-fix files, and the
  fixed-label targets are likewise empty. Every gate that would authorise a
  ranking still fails on them.
- Any second opinion on run #2b rows beyond the round's hand-check; whether the
  rule-ambiguous "whale" row is correct or incorrect (a human call, queued).
- Semantic label error above the 5.0% floor: the code sweep cannot see
  paraphrase/synonym false negatives, so the true rate is bounded below only.
- T4/fp16 determinism and batch invariance (D27); run #3's base rate at 3B
  scale; a second subject model; run #2's per-dataset intervals (lost).
- Measured per-row labelling time (three rows would instrument it).
