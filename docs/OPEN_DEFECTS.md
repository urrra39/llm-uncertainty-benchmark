# Open defects

Generated from `OPEN_DEFECTS` in `scripts/audit_docs.py` — edit the
source, not this file. The docs audit fails when the two disagree.

| ID | Title | Status | What closes it | What it blocks |
|---|---|---|---|---|
| D27 | Batched generation perturbs per-token logprobs (2.52e-02, CPU/bfloat16) | open | `unc-bench nondeterminism` batch_invariance.pass == true on the run's target device and dtype (T4, fp16 for run #3) | configs/run3_gpu.yaml generation_batch_size (pinned at 1) |
| D36 | No per-config lock file; concurrent generate halves throughput | open | second invocation against the same config refuses to start; covered by a test that launches two runs | operator time, not correctness |
| HUMAN-COVERAGE | No human has verified any label (gates fail at 0.0) | open | PRE-run labeling_protocol_validated at >= 0.50 on data/human_validation_sample.csv, then POST-run human_label_coverage at >= 0.80 on the new run's subsample, per the ordering in docs/PREREGISTRATION.md | validity_gates.all_passed for every future run |
| RUN2-ARTIFACTS | Run #2 per-row artifacts lost to the old ignore policy | permanent | none recoverable; run #3 artifacts are tracked | run #2 per-dataset bootstrap intervals (Hanley-McNeil fallback stands) |
| GEN-DETERMINISM | Generation reproducibility unmeasured on every run | open | `unc-bench nondeterminism` greedy double-run mismatch rate on the run's rows | no claim in either direction (README states this) |
