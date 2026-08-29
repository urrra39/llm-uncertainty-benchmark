# Limitations

The run described in the README completed. These are the reasons not to trust
its ranking.

## 1. The sample is too small to separate 21 signals

n=100 questions, of which 75 were answered and 7 were correct. AUROC is
computed against those 7 positives. The widest confidence interval in the
table is 0.500 wide and the narrowest is 0.208; the gap between first and
second place is 0.007. Nothing in the table is separated from anything else,
and the README says so directly.

The controlling evidence is the random baseline. A uniform random score drew
AUROC 0.746 [0.590, 0.876] on these rows. It should be 0.5. Twelve of the
twenty real signals have point estimates inside that interval. Any statement
of the form "family A beats family B here" is a statement about which way the
noise fell.

## 2. The subject model is 14x smaller than intended

The design targets Qwen2.5-7B-Instruct on vLLM. This machine has no GPU and
the available API gateway returns no `logprobs` for any model it allows, which
rules out a hosted subject model entirely — family A is the whole point of the
comparison and it needs token logprobs. So the subject is
Qwen2.5-0.5B-Instruct on 2 CPU cores.

The consequence is not just noise, it is a different regime. A 0.5B model gets
7 of 75 committed trivia answers right. The uncertainty-signal literature this
study is testing was developed on models that are right most of the time, where
the interesting question is which of the minority of errors a signal catches.
Here the model is wrong 90.7% of the time it answers, so there is very little
correct-answer mass for any signal to identify. Results should not be
extrapolated to a 7B or 70B model.

## 3. The base error rate makes selective prediction useless

Base rate of error among answered rows is 0.907. Coverage at the 90%-accuracy
target is 0.013 — one question — for the two signals that reach it at all, and
undefined for the other nineteen. The best threshold I could find on the
best signal takes error from 0.907 to 0.714 at 19% coverage. That is a real
improvement and it is not a shippable system. The risk-coverage figure shows
curves that stay above 0.62 error at every coverage above 11%.

## 4. DeLong and AUPRC were not computed

The plan called for pairwise DeLong tests with Holm-Bonferroni correction and
AUPRC alongside AUROC. Neither is implemented in the analysis layer, and
building them this session was out of scope: the session's mandate was to
execute the run, not to extend the apparatus. Pairwise separation is therefore
argued from confidence-interval overlap, which is the more conservative test —
it cannot manufacture significance that DeLong would have denied, but it is
less sensitive, so "no significant winner" is stated on weaker evidence than
intended. The AUPRC column is absent rather than estimated.

The logistic-regression combination of two signals was also skipped. It
answers a different question than the title asks.

## 5. Judge agreement is perfect, which is itself a caveat

Cohen's κ = 1.000 between gpt-5-mini and claude-haiku-4-5 on all 71 judged
rows. That is not evidence of a well-calibrated judging setup; it is evidence
that the task was easy. 64 of 71 judged rows are incorrect and most are
unambiguously so. Two judges agreeing that "The Wizard of Oz" is not "The
Third Man" tells you very little about how they would handle a genuinely
borderline alias match, which is the case the judge exists for.

Related: `claude-haiku-4-5` ignored the "reply with exactly one word"
instruction on 45 of 71 items and answered with the verdict followed by an
unsolicited justification. The original strict parser discarded all of those,
which silently reduced the second judge to 15 usable verdicts. Had I not
inspected the cache I would have published a κ computed on a
compliance-selected subset and called it agreement on the run. The parser now
accepts a leading verdict.

## 6. No human validation

`data/human_validation_sample.csv` contains all 100 rows with an empty
`human_label` column. Nobody labeled it. There is no measured human-judge
agreement, so the entire label set rests on two LLM judges that agree with
each other and were never checked against a person.

## 7. Single dataset, single prompt, single seed

TriviaQA only. PopQA and SimpleQA are not in this run, so nothing here speaks
to whether the ranking is stable across question distributions — and given
point 1, it would not be detectable if it were not. One fixed prompt, one
greedy decode at seed 0, one sampling seed base. The nondeterminism probe
exists in the CLI and was not run this session, so CPU-level run-to-run drift
in the greedy answers is unquantified for this run.

## 8. Abstention handling is a judgment call

25 of 100 rows are `UNKNOWN` and are excluded from the discrimination task.
This is defensible — a refusal is not a factual error, and it is trivially
predictable from the logprob of a token the model was told to emit, so
including it would inflate every signal. But it is a choice, and it removes a
quarter of the data. The `with_abstentions` view in `results.json` reports the
alternative treatment for anyone who disagrees: there the ranking changes at
the top (total logprob leads at 0.886) and the conclusion does not — the
intervals still all overlap.
