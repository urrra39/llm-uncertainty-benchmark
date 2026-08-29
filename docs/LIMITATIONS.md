# Limitations

## The run has not been executed

This is the limitation that subsumes the rest. The measurement apparatus is now
complete — all four signal families, the labeling pipeline, 217 tests — and the
harness that would drive it is not. The five pipeline stages, the analysis module
and the figures are unwritten, so `make all` does not run end to end, there is no
`results.json`, and there is no AUROC table anywhere in this repository.

Nothing here should be cited as a result. Specifically absent: every AUROC, the
bootstrap CIs, the DeLong comparisons, the risk-coverage curves and AURC, the ECE
and reliability diagrams, the Platt-scaled calibration, the correlation heatmap,
the per-dataset breakdown, the N=1/2/3/5 ablation, and the 2-feature logistic
regression. Each of those was specified and none was run.

I also did not follow the brief's instruction to stop building at 50% of budget
and start running. The signal families and labeling ran long, partly because
three latent bugs only surfaced on first contact with real weights and a live
gateway (DECISIONS D29-D31). The result is that the wrong half of the project is
finished: the part that produces numbers is tested and the part that would
collect them does not exist.

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

Re-measured this session on the real model rather than inherited: 6.6 s for a
greedy answer including torch warmup, 8.0 s for five sampled continuations in one
batched call, ~2.3 s for a verification forward pass. Roughly 20-35 s of CPU per
question end to end.

The harder constraint is memory, and it was only found by measuring. Peak
resident set with the generator alone is 1.57 GB of 2.0 GB available. The NLI
model cannot be co-resident with it, so family B cannot be scored inline during
generation and the pipeline has to load the two models in separate stages. Any
future implementation of the stages has to respect that; a design that holds both
at once will be OOM-killed rather than merely slow.

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

## Signals are unvalidated against outcomes

Every signal is unit-tested for correctness of its arithmetic and for its
orientation. None has been checked against a single real correctness label,
because no labels exist. A signal can be arithmetically perfect and carry no
information about whether an answer is wrong, and that is exactly the question
the project was built to answer.

## Verbalized confidence expected to be degenerate

The design predicts that a 0.5B model reports a near-constant confidence
(typically 90) regardless of whether it is right. That prediction is untested
here. It is recorded as an expectation rather than a finding, and the signal is
retained rather than dropped so the degeneracy would show up in the results if it
is real.

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
