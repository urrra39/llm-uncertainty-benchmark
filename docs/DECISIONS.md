# Decisions

Running log. Newest entries at the bottom of each section. Every non-obvious
choice gets one line of rationale so I can argue with myself later.

## Environment probe (before any code)

I probed the machine before committing to a model path, because the whole design
hinges on whether token logprobs are available.

| Check | Result |
| --- | --- |
| `nvidia-smi` | not found, no GPU |
| RAM | 2.0 GB total, ~1.7 GB available |
| CPU | 2 cores |
| Disk | 20 GB free |
| `gpt-4o-mini` via available API | 400, model not in the allow-list |
| `logprobs=true` on every allowed API model | field absent from the response body |

The API gateway I have accepts `logprobs: true` and `top_logprobs: 5` without
error but returns no `logprobs` object at all. I checked this at the raw HTTP
level, not just through the SDK, to rule out client-side stripping. Without
token logprobs, signal family A is impossible, and family A is half the point of
the project.

- **D1. No GPU, so vLLM + Qwen2.5-7B-Instruct is out.** The brief's stated
  fallback is gpt-4o-mini. That model is not available on my gateway, and no
  available model returns logprobs. Both prescribed paths are dead.
- **D2. Third tier: run a small model locally on CPU via HuggingFace
  transformers.** Chosen model `Qwen/Qwen2.5-0.5B-Instruct`, same family and
  tokenizer lineage as the intended 7B, so prompt formatting and the True-token
  assertions carry over unchanged if a GPU appears. Measured: 0.59 GB resident
  in bfloat16, ~0.9 s per greedy 12-token answer, 5 sampled continuations in one
  batched call in ~14 s. `output_scores=True` gives exact per-token logprobs and
  full-vocabulary distributions, which is strictly more than the API's top-5
  would have given.
  - Cost of this choice: a 0.5B model is much weaker than a 7B, so the absolute
    error rate will be high and the accuracy numbers are not comparable to
    published 7B results. The *ranking of signals* is the object of study, and
    that is still measurable. Recorded in LIMITATIONS.
- **D3. fp32 OOMs, use bfloat16.** First attempt loaded the 0.5B in fp32 and the
  kernel OOM-killed the process at 1.87 GB resident. bfloat16 halves that.
- **D4. Anthropic models are not the subject under test.** The Anthropic API
  does not expose token logprobs at any tier, so signal family A cannot be
  computed for a Claude model. This is a property of the API surface, not a
  judgement about the model. Several Claude models are available on my gateway
  and are used as *judges*, where only text output is needed.
- **D5. Model choice stays behind one interface.** `ModelClient` is a protocol
  with two implementations, local-transformers and OpenAI-compatible HTTP. The
  config selects one. If a GPU becomes available, switching to vLLM + 7B is a
  YAML edit, not a code change.

## Engineering

- **D6. `requires-python == 3.11.*` pinned exactly.** The sandbox has 3.13 and CI
  runs 3.11; pinning stops me from accidentally depending on 3.12+ syntax that CI
  would reject.
- **D7. Heavy deps (`torch`, `transformers`) live in an optional `local` extra.**
  CI never installs them. Every test reads committed fixtures, so CI needs no GPU,
  no model download and no network.
- **D8. mypy `strict = true` from the first commit.** Cheaper than retrofitting.
- **D9. NLI model: `MoritzLaurer/DeBERTa-v3-base-mnli`, not `-large`.** The brief
  asks for deberta-v3-large-mnli. large is ~1.6 GB in fp32 and would not
  co-reside with the generator in 2 GB. base measured at 0.73 GB resident and
  0.09 s per pair batched. This is the documented downgrade path in the failure
  policy. Label order verified from `config.id2label`:
  `{0: entailment, 1: neutral, 2: contradiction}` — I read it off the config
  rather than assuming, because several MNLI checkpoints on the Hub use the
  reverse order and that would silently invert every clustering decision.
