# Open defects

Generated from `OPEN_DEFECTS` in `scripts/audit_docs.py` — edit the
source, not this file. The docs audit fails when the two disagree.

| ID | Title | Status | What closes it | What it blocks |
|---|---|---|---|---|
| D27 | Batched generation perturbs per-token logprobs (2.52e-02, CPU/bfloat16) | open | `unc-bench nondeterminism` batch_invariance.pass == true on the run's target device and dtype (T4, fp16 for run #3) | configs/run3_gpu.yaml generation_batch_size (pinned at 1) |
| D36 | No per-config lock file; concurrent generate halved throughput | closed by per-config lock with stale takeover (tested) | second invocation against the same config refuses to start; covered by a test that launches two runs | nothing further |
| HUMAN-COVERAGE | No human has verified any label (gates fail at 0.0) | open | PRE-run labeling_protocol_validated at >= 0.50 on data/human_validation_sample.csv, then POST-run human_label_coverage at >= 0.80 on the new run's subsample, per the ordering in docs/PREREGISTRATION.md | validity_gates.all_passed for every future run |
| RUN2B-LABELSET | Run #2b's published table is computed from the corrected label set (data/run2b/labels_fixed.parquet -> results_run2b_fixedlabels.json); the pre-fix containment-rule set is archived as the comparison (results_run2b.json) | code sweep closed (relabel committed and primary); human-coverage half open, tracked by HUMAN-COVERAGE | the code sweep's six demonstrable errors are corrected in the published label set; what remains open is the HUMAN-COVERAGE gate: >= 0.80 human coverage of the run's fixed-label fuzzy-decided rows (data/fuzzy_decided_rows_fixed.csv, docs/HUMAN_LABELING.md) | nothing further on the code-sweep half; validity_gates.all_passed for the run still waits on the human gates (HUMAN-COVERAGE) |
| RUN2-ARTIFACTS | Run #2 per-row artifacts lost to the old ignore policy | permanent | none recoverable; run #3 artifacts are tracked | run #2 per-dataset bootstrap intervals (Hanley-McNeil fallback stands) |
| GEN-DETERMINISM | Generation reproducibility unmeasured on every run | open | `unc-bench nondeterminism` greedy double-run mismatch rate on the run's rows | no claim in either direction (README states this) |
