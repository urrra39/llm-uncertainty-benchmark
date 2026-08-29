# llm-uncertainty-benchmark

**Which cheap uncertainty signal best predicts that an LLM's short-form factual answer is wrong?**

Status: scaffold. The results table, figures and reproduction instructions land
as the five pipeline stages are implemented. What is fixed already:

- All signals are computed on the **same greedy (temperature=0) answer**.
- Three families are compared at very different costs: token logprobs (1x),
  self-consistency over 5 extra samples (5x), self-verification (~1.1x).
- Trivial baselines (answer length, question length, random) are reported beside
  every signal. If a sophisticated signal fails to beat answer length, this
  README says so at the top.
- Metric: AUROC for predicting **INCORRECT**, with 95% stratified bootstrap CIs.

See `docs/DECISIONS.md` for the running log of choices and their rationale.

## Layout

```
src/unc_bench/          package
  datasets/             one builder per dataset
  signals/              families A, B, C and the trivial baselines
  analysis/             AUROC, DeLong, risk-coverage, calibration
configs/                YAML configs validated by Pydantic
tests/                  fixture-driven, no live model calls
docs/                   DECISIONS.md, LIMITATIONS.md
```

## Development

```bash
make setup     # pinned deps via uv, Python 3.11
make check     # ruff + mypy strict + pytest
```

## License

MIT.
