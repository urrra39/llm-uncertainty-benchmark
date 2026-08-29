# Limitations

## The run has not been executed

This is the limitation that subsumes the rest. The repository contains the
measurement apparatus for the five stages that are implemented, and no
measurements. Nothing here should be cited as a result.

## Model specificity

The design targets Qwen2.5-7B-Instruct on vLLM. This machine has no GPU, and
the API gateway available to it returns no `logprobs` object for any model it
allows, which rules out both the primary path and the brief's gpt-4o-mini
fallback (see docs/DECISIONS.md D1). The configured subject is therefore
`Qwen2.5-0.5B-Instruct` on CPU.

Two consequences. A 0.5B model is much weaker than a 7B, so its absolute error
rate would be far higher than published 7B numbers and not comparable to them.
More importantly for the research question, it is not established that the
*ranking* of uncertainty signals is stable across model scale. If the ranking
were measured at 0.5B, it would be a claim about 0.5B until someone reruns it at
7B. The config makes that rerun a YAML edit.

## Anthropic models are not the subject

The Anthropic API does not expose token logprobs, so signal family A cannot be
computed for a Claude model. That is a property of the API surface. Claude
models are used as label judges here, where only text output is needed.

## Short-form only

Every signal and every label assumes the answer is a short factual span that can
be compared by normalized exact match or adjudicated by a judge in one call.
None of this transfers to long-form generation, where correctness is not binary
and there is no single answer span over which to average logprobs.

## Compute ceiling

Measured on this machine: ~1 s per greedy answer, ~14 s for a batch of 5 samples,
~2.8 s per verification forward pass, ~0.09 s per NLI pair. That is roughly 35 s
of CPU per question, so the 1200-question run is on the order of 12 hours on 2
cores before judging. The failure policy's fallback of cutting to 600 questions
proportionally does not bring this into range either.

## Automated rather than human label validation

The design validates labels with a second independent judge and reports Cohen's
kappa, and writes 100 rows to `data/human_validation_sample.csv` with an empty
`human_label` column. That is agreement between two automated judges, not
accuracy against human labels. Two judges can agree and both be wrong,
especially on alias-heavy PopQA items where the boundary between a correct
alias and a near-miss is genuinely unclear.

## NLI downgrade

`deberta-v3-large-mnli` does not fit alongside the generator in 2 GB, so the
config specifies `deberta-v3-base-mnli`. The base model is weaker at the
bidirectional entailment judgements that semantic-entropy clustering depends on,
which would add noise to that one signal specifically.

## Batch non-determinism, unmeasured

The README is supposed to report the disagreement rate over 50 greedy prompts run
twice. The `nondeterminism` stage is specified in the Makefile and not yet
implemented, so this number does not exist. Note that the relevant effect on a
CPU transformers backend differs from vLLM's: vLLM's non-determinism comes from
batch-dependent kernel scheduling, which this backend does not do.

## Dataset choice

PopQA, TriviaQA and SimpleQA are all English, all entity-centric, and all drawn
from Wikipedia-adjacent sources. A signal that works here may not work on
domain-specific factual questions with different popularity distributions.
