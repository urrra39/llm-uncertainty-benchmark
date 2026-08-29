# llm-uncertainty-benchmark

**Which cheap uncertainty signal best predicts that an LLM's short-form factual
answer is wrong?**

## There are no results in this repository yet

I want that at the top rather than buried, because the natural shape of a README
like this one is a results table, and a fabricated table would defeat the whole
point of the exercise.

What exists is the measurement apparatus for five of the pieces, all green under
`make check`. What does not exist is a single benchmark number. No run has been
executed. See [docs/DECISIONS.md](docs/DECISIONS.md) for why, and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what that invalidates.

The short version of why: the project needs token logprobs, and this machine
cannot get them. No GPU, so vLLM with Qwen2.5-7B-Instruct is out. The brief's
fallback, gpt-4o-mini, is not on the API gateway available here, and every model
that *is* available returns no `logprobs` object at all. I checked that at the
raw HTTP level, not just through the SDK. The configured path is therefore
Qwen2.5-0.5B-Instruct running on CPU through HuggingFace transformers, which does
give exact per-token logprobs, at about 35 s of CPU per question once sampling
and verification are included. 1200 questions is ~12 hours on 2 cores. That did
not fit in the session.

## The design

One question, answered by measurement: for a fixed greedy answer, which cheap
signal best ranks it as likely wrong?

Every signal is computed on the **same** greedy (temperature=0) generation. That
constraint is enforced in the config schema, not just documented: `GreedySpec`
refuses a non-zero temperature, so family A cannot be computed on a sampled
generation by misconfiguration.

| Family | Cost | Signals |
| --- | --- | --- |
| A. Token logprobs | 1x | mean logprob over the answer span, min logprob, perplexity, mean top-5 predictive entropy, first-answer-token logprob, length-normalized total |
| B. Self-consistency | 5x | exact-match agreement rate, distinct normalized answers, mean pairwise token F1, semantic entropy via bidirectional-entailment clustering, length-normalized semantic entropy |
| C. Self-verification | ~1.1x | P(True) from logprobs, P(True) with the 5 samples in context, verbalized 0-100 confidence |
| T. Trivial | 0x | answer length, question length, random |

The trivial baselines are the point of the table, not a footnote. Short answers
mechanically have higher mean logprob, so any family-A result that is not
reported beside the answer-length baseline is uninterpretable. If a sophisticated
signal fails to beat answer length, this README is supposed to lead with that.

Metric is AUROC for predicting **incorrect** (positive class = incorrect, every
signal oriented so higher means more likely wrong), with 95% stratified bootstrap
CIs over 10,000 resamples.

Datasets: PopQA 600, TriviaQA rc.nocontext validation 400, SimpleQA 200. All
three verified reachable and their schemas confirmed. `UNKNOWN` is treated as
abstention and reported separately, because counting abstentions as errors
inflates every AUROC.

## What is implemented

Green under ruff, ruff format, mypy strict and pytest on Python 3.11 with no GPU.
84 tests, all fixture-driven, no live model calls.

- Config schema (`src/unc_bench/config.py`) with validators that block specific
  mistakes: non-greedy family A, sampling at temperature 0, an ablation N above
  the sample count, a judge that is the model under test, prompt templates
  missing their placeholders.
- Model client (`src/unc_bench/client.py`). One `Protocol`, two backends. The
  answer span only is scored, never prompt tokens. Raw per-token top-k
  distributions are persisted, so a new family-A signal can be added later
  without regenerating.
- Response cache (`src/unc_bench/cache.py`). SHA-256 over backend, model,
  rendered prompt, temperature, top_p, seed, max tokens, top_logprobs and n.
  A rerun of a finished stage issues zero calls.
- Normalization (`src/unc_bench/normalize.py`). Exact match, token F1,
  abstention detection.
- Dataset builder base plus the PopQA builder.

## What is not implemented

TriviaQA and SimpleQA builders, the five pipeline stages, all three signal
families, the three-stage labeling pipeline with inter-judge kappa, and the
analysis and figure code. The Makefile targets for these exist and will fail.

## Two things that did not go as planned

The corruption test on the response cache failed for a reason I had not
anticipated. I caught `OSError`, `JSONDecodeError` and `EOFError` around the
gzip read, on the assumption that those covered a truncated file. A truncated
gzip member actually raises `zlib.error`, which inherits from `Exception` and not
from `OSError`, so a half-written cache entry propagated instead of reading as a
miss. That would have surfaced as a crash mid-run after an interrupt, which is
precisely when a resumable pipeline is supposed to work.

The normalization tests failed three ways, and only one was the code. The real
bug: I used `string.punctuation`, which is ASCII-only and does not contain U+2019
RIGHT SINGLE QUOTATION MARK. `NFKC` does not fold that to an ASCII apostrophe
either, so the two spellings of O'Brien normalized to different strings, and the
curly form is what models actually emit. The table now covers every codepoint in
Unicode category P. The other two failures were my own bad fixtures: I used
single letters as filler tokens, and a bare "a" is an article that normalizes
away, so my hand-computed F1 values were wrong rather than the function.

## Reproducing what exists

```bash
git clone https://github.com/urrra39/llm-uncertainty-benchmark
cd llm-uncertainty-benchmark
make setup
make check        # ruff + ruff format + mypy strict + pytest
```

`make setup EXTRA=local` additionally installs torch and transformers. CI
installs neither.

CI runs on every push to `main` from
[`.github/workflows/ci.yml`](.github/workflows/ci.yml): ruff, ruff format, mypy
strict and pytest on Python 3.11, with `UNC_BENCH_OFFLINE=1` so no test can
reach the network. `make check` runs the identical set locally.

Hardware this was developed on: 2 vCPU, 2.0 GB RAM, no GPU, Python 3.11.

## Model and dataset cards

- Subject configured: `Qwen/Qwen2.5-0.5B-Instruct`, bfloat16 on CPU, greedy
  decoding, seed 0, 24 max new tokens. Intended subject:
  `Qwen/Qwen2.5-7B-Instruct` on vLLM. Same family, so prompt formatting and the
  single-token True/False assertions carry over.
- NLI: `MoritzLaurer/DeBERTa-v3-base-mnli`. Downgraded from `-large`, which does
  not fit in 2 GB alongside the generator. Label order read off `config.id2label`
  rather than assumed, since several MNLI checkpoints on the Hub reverse it.
- Judges: `gpt-5-mini` primary, `claude-haiku-4-5` secondary. Both need text
  output only, so the gateway's missing logprobs do not matter for judging.
- PopQA: 14k Wikidata-triple questions with popularity counts, read from
  `test.tsv`. Long-tail entities, which is why it is here.
- TriviaQA `rc.nocontext` validation: 17,944 rows, gold aliases in
  `answer.normalized_aliases`.
- SimpleQA: 4,326 rows, adversarially collected against stronger models.

## Inter-judge kappa

Not measured. The labeling pipeline is not implemented.

## License

MIT.
