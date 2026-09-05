# Audit response: item-by-item disposition

Each item of the external audit maps to the commit that addressed it or to an
explicit reasoned refusal. Silence is not a deliverable; disagreement with
evidence is.

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

# Round 4: closing the audit gap (in progress)

## Withdrawal bound

Worst case from committed artifacts: 14 alias-decided echo rows observed in
100 plus all 20 unobserved rows, so up to **34 of 120 labels (28%)** could
flip. That bound exceeds every interval in the run and withdraws the ranking
(see README "Run #2 and why its ranking is withdrawn"). Item-by-item
dispositions for round 4 follow as each part lands.
