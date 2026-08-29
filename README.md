# llm-uncertainty-benchmark

**Which cheap uncertainty signal best predicts that an LLM's short-form factual
answer is wrong?**

## There is no results table in this repository yet

That belongs at the top rather than buried. The natural shape of a README like
this one is an AUROC table, and a fabricated table would defeat the entire point
of the exercise.

What exists now is the complete measurement apparatus: all four signal families
(21 signals), the labeling pipeline with Cohen's kappa, and 217 tests, green
under `make check` and CI. What does not exist is a single benchmark number. The
five pipeline stages that would drive the apparatus are not written, so no run
has been executed. See [docs/DECISIONS.md](docs/DECISIONS.md) for the full
account and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what that invalidates.

Two things are now verified against reality rather than assumed, which is the
main thing this session added beyond code:

The subject model produces exact per-token logprobs on this machine.
`Qwen2.5-0.5B-Instruct` in bfloat16 answers "Who wrote Dracula?" with "Jules
Verne" (wrong, which is the point of using a 0.5B model for an error-prediction
benchmark) in 6.6 s, and five sampled continuations come back in one batched
call in 8.0 s. Peak resident memory is 1.57 GB of the 2.0 GB available.

Both judges work. `gpt-5-mini` and `claude-haiku-4-5` each returned the bare
verdict word on all three probe items, correctly grading "Bram Stoker" as
CORRECT, "Jules Verne" as INCORRECT, and "Bram Stoker, the Irish novelist" as
CORRECT — the last being the exact-match miss that motivates having a judge at
all. A real inter-judge kappa is obtainable here. It has not been measured,
because measuring it requires the labeling stage that drives the module.

The reason the subject is a 0.5B CPU model is a stated scope limit, not an
apology: this machine has no GPU, and the API gateway returns no `logprobs`
object for any model it allows, checked at the raw HTTP level. Local transformers
inference is the only path to token logprobs here, and family A is half the
research question.

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

Green under ruff, ruff format, mypy strict and pytest on Python 3.11 with no
GPU. 217 tests, all fixture-driven, no live model calls in the suite.

| Module | What it does |
| --- | --- |
| `config.py` | Pydantic YAML schema. Validators block non-greedy family A, sampling at temperature 0, an ablation N above the sample count, a judge that is the subject model, and prompt templates missing placeholders. |
| `client.py` | One `Protocol`, two backends. Answer span only is ever scored. Raw per-token top-k distributions are persisted, so a new family-A signal needs no regeneration. |
| `cache.py` | Content-addressed gzip cache, SHA-256 over ten fields. A rerun of a finished stage issues zero calls. |
| `normalize.py` | Exact match, token F1, abstention detection, full Unicode punctuation stripping. |
| `signals/base.py` | The signal registry. Orientation is declared here and nowhere else. |
| `signals/logprob_signals.py` | Family A, 9 signals. |
| `signals/nli.py` + `consistency.py` | Family B, 6 signals, bidirectional-entailment clustering and semantic entropy. |
| `signals/verification.py` | Family C, 3 signals, P(True) renormalized over the True/False pair. |
| `signals/trivial.py` | Family T, 3 baselines. |
| `labeling.py` | Abstention routing, exact match, judge rubric, strict verdict parsing, heuristic fallback, Cohen's kappa. |
| `datasets/` | Builder base and the PopQA builder. |

Three details in there are worth more than the feature list.

Orientation lives in exactly one place. Every signal declares whether its raw
value rises with confidence or with risk, and one function applies the flip, so
the exported number always means "higher = more likely wrong". An inverted signal
does not fail loudly — it reports AUROC 1-x, so a genuinely strong signal at 0.72
reads as 0.28 and nothing in the table points at the cause. There is a test per
family that asserts the sign of every signal at once.

P(True) is renormalized over the {True, False} pair rather than read as
`exp(logprob(" True"))`. Measured on the real model with the real prompt, the raw
value is 0.0000 and the renormalized value is 0.0076. The raw number is not a
small probability of correctness; it is an artifact of most of the next-token
mass going to "Yes" or a newline. On Qwen2.5 the tokens resolve to `" True"` =
3007 and `" False"` = 3557, read off the tokenizer rather than assumed, and the
resolver raises if the two ever collide on one id — that collision would pin
P(True) at exactly 0.5 for every item and read as "self-verification is
uninformative" rather than as a broken lookup.

Abstention is its own label category. A refusal is trivially predictable from any
of these signals, so counting refusals as errors would inflate every AUROC by
measuring a tautology.

## What is not implemented

The five pipeline stages and their CLI wiring, the analysis module (AUROC,
stratified bootstrap, DeLong, calibration, risk-coverage), the figures, and the
TriviaQA and SimpleQA builders. The Makefile targets for these exist and will
fail. Because the stages are absent, `make all` does not run end to end and there
is no `results.json`.

## Four things that did not go as planned

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

Three latent bugs surfaced the moment previously-untested code met real weights
and a real gateway, and all three were in modules that had been passing their
fixture tests for a session. `client.py` passed `dtype=` to
`from_pretrained`; transformers 4.46 has no such parameter and raises
`TypeError` from the model constructor. The DeBERTa-v3 tokenizer needs
`protobuf` to convert its sentencepiece model, so the NLI model raised
`ImportError` while the BPE-backed Qwen tokenizer loaded fine — making the
failure look unrelated to the NLI downgrade that caused it. And openai 1.54.4
constructs its HTTP client with `proxies=`, which httpx removed in 0.28.0, so
`OpenAI()` raised before issuing a single request. None of these are visible from
reading this repository's code, and none of them were caught by 84 passing tests.

Two of my own tests were wrong rather than the code, again. I asserted that the
Wu-penalized total and the mean logprob must rank a pair differently, and picked
a pair where both signals agree — dividing a fixed negative total by anything
larger moves it the same direction, so the fixture could not have separated them.
And I used "a b" / "c d" as filler strings in a clustering test, where "a" is an
article that normalizes away, so my scripted entailment scores were keyed on
pairs the code never asked about. That is the same fixture trap that cost me two
normalization tests last session, which suggests the lesson has not stuck.

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

Not measured. The labeling module is implemented and both judges are verified
working against the live gateway, but computing kappa requires the labeling
stage, which is one of the five unwritten pipeline stages. The module has a
`cohens_kappa` that returns NaN with `trustworthy=False` rather than 0.0 when two
judges are unanimous, because 0/0 is unanimity with no variance to measure and
reporting 0.0 would read as total disagreement.

## License

MIT.
