# Which cheap uncertainty signal best predicts an LLM's factual errors?

> **Run #2's ranking below is withdrawn.** Up to 34 of its 120 labels (28%)
> could flip under the echo-contamination bound beside the tables, which
> exceeds what any interval here can absorb. The tables stay as the record of
> the invalidated run, on the same footing as run #1. The primary result is
> run #2b's, once it has run.

> Status, generated from `results_run2b.json` (`scripts/render_readme_header.py`):
> Primary run: run2b_clean (n=120, 71 incorrect / 49 correct).
> VALIDITY FAILED: labeling_protocol_validated, human_label_coverage as recorded in this file (t_random 0.523 [0.416, 0.628]).
> Label quality: 0/100 human-labelled — the correctness of the label set is unmeasured.

## Run #2b (primary, pending the human gate)

Run #2b re-ran run #2's exact configuration (Qwen2.5-0.5B-Instruct, same
decoding, prompts, sampling, NLI, n=120) with the decontaminated dataset:
no `capital of`, gold-in-question rows dropped, near-duplicates deduped, and
a balanced 60/60 split. Pre-registered before execution
(`docs/PREREGISTRATION.md`, "Run #2b" section). 27 of 27 registered signals
scored — the divergence is closed with real numbers.

Two honest caveats travel with every number below. First, **labels are
heuristic**: no judge credentials exist in an offline environment, so 47 rows
settled by exact match and 73 by the fuzzy containment rule, with no kappa.
Second, **the human gates fail**: `labeling_protocol_validated` and
`human_label_coverage` both read 0.0, so by the project's own rule the ranking
is not yet publishable. It is shown here as a measurement with that status
attached, not as a finding. `data/human_validation_sample_run2b.csv` (100
rows, 49 correct / 51 incorrect, `human_label` empty) is ready for the hand
labelling that opens the gates.

The three classical gates pass: `t_random` scores 0.445 [0.296, 0.598] on
PopQA and 0.571 [0.384, 0.752] on TriviaQA (both contain 0.50), classes are
71/49 overall, abstentions 0. Base rates: PopQA 23/37 incorrect (38%),
TriviaQA 48/12 (80%).

### Run #2b per-dataset AUROC (60/60, real bootstrap intervals)

| signal | PopQA (23/37) | TriviaQA (48/12) | stratified |
|---|---|---|---|
| `b_disagreement_rate` | 0.768 [0.646, 0.882] | 0.727 [0.592, 0.847] | 0.747 |
| `a_total_logprob` | 0.825 [0.711, 0.922] | 0.656 [0.514, 0.790] | 0.741 |
| `a_length_normalized_logprob` | 0.811 [0.690, 0.914] | 0.656 [0.516, 0.790] | 0.734 |
| `b_distinct_count` | 0.742 [0.619, 0.857] | 0.712 [0.567, 0.842] | 0.727 |
| `b_mean_pairwise_f1` | 0.751 [0.627, 0.868] | 0.666 [0.495, 0.821] | 0.708 |
| `b_disagreement_rate_samples_only` | 0.729 [0.604, 0.850] | 0.673 [0.504, 0.826] | 0.701 |
| `a_mean_logprob` | 0.740 [0.609, 0.861] | 0.490 [0.319, 0.661] | 0.615 |
| `c_p_true_plain` | 0.795 [0.660, 0.912] | 0.528 [0.311, 0.733] | 0.662 |
| `t_question_length` | 0.499 [0.392, 0.611] | 0.553 [0.371, 0.730] | 0.526 |
| `c_verbal_confidence` | 0.504 [0.368, 0.639] | 0.547 [0.383, 0.707] | 0.526 |
| `t_random` | 0.445 [0.296, 0.598] | 0.571 [0.384, 0.752] | 0.508 |

### What decontamination did (the E4 prediction, tested)

- **`a_mean_logprob` on PopQA: 0.514 → 0.740, clearing chance
  [0.609, 0.861].** Predicted up; confirmed strongly. The confident stratum
  was noise in run #2 and signal here.
- **`c_verbal_confidence` on PopQA: 0.395 → 0.504.** Predicted up; confirmed
  in direction only — it sits at chance, not above it. The hypothesis is
  right about the mechanism and overclaims nothing about this signal.
- **`t_question_length` stripped of provenance:** pooled 0.684 in run #2,
  0.499 on PopQA and 0.526 stratified here. It never measured uncertainty.
- **`t_random` is now significantly worse than the leader** (−0.276,
  p_holm = 0.0076 over 21 distinct comparisons, 5 worse-significant total).
  Run #2's non-rejection was power (n=120, Holm over 20), not a defective
  test — the B1 calibration and this movement agree.
- **Pooled leader is family A** (`a_total_logprob` 0.799), with no winner
  (gap 0.008, CIs overlap). The samples-only variants trail their
  greedy-included twins by 0.02–0.04, so the temperature-mixing bias is small
  on this run but measured rather than assumed.
- **N-ablation does not saturate at N=3 here:** 0.641 / 0.700 / 0.742 / 0.765
  at N=1/2/3/5, with N=1 significantly below N=5 (−0.124, p_holm = 0.016)
  and N=3 vs N=5 indistinguishable (−0.023, p = 0.698). Run #2's "use N=3"
  survives as cost advice, not as an optimum.
- **Clustering audit:** 1 disagreement in 16 audited rows (Wilson
  [0.011, 0.283]); family-B token multiplier measured at 6.01× for 6.0×
  calls — the call-count price was honest.

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
