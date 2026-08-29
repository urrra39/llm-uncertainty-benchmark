"""Stage 2: every model call that needs the generator loaded.

One question costs, in order: a greedy answer with top-5 logprobs, N sampled
answers, two verification forward passes, and one verbalized-confidence
generation. That is everything families A, C and T need. Family B needs the NLI
model instead, and it is a separate stage for a measured reason: the generator
alone peaks at 1.57 GB of the 2.0 GB available, so loading DeBERTa alongside it
gets the process killed. See docs/DECISIONS.md D9.

Resumable at question granularity. Rows already in the checkpoint are skipped,
and the checkpoint is rewritten every `flush_every` questions, so an interrupt
costs at most that many questions of work — and usually nothing at all, because
the response cache already holds the replies.

The verification prompts are rendered with an EMPTY system message on purpose.
The run's system prompt instructs the model to answer with the shortest factual
span and to say UNKNOWN when unsure, which is the right instruction for answering
and the wrong one for grading: reused here it pushes probability mass toward
UNKNOWN at exactly the position where P(True) is read off the True/False pair.
"""

from __future__ import annotations

import gc
from typing import Any

import pandas as pd

from unc_bench.cache import ResponseCache
from unc_bench.client import (
    CachedClient,
    _generation_from_dict,
    _generation_to_dict,
    build_client,
)
from unc_bench.config import Config
from unc_bench.datasets.base import frame_to_questions
from unc_bench.normalize import clean_model_answer
from unc_bench.signals.verification import (
    TrueFalseTokens,
    parse_verbal_confidence,
    resolve_true_false_tokens,
    score_p_true,
)
from unc_bench.stages.common import (
    Progress,
    StagePaths,
    done_qids,
    json_str,
    merge_timings,
    read_checkpoint,
    write_checkpoint,
)
from unc_bench.types import Generation, Question, VerificationResult

# How many questions between checkpoint rewrites. Not a result-affecting number,
# so it stays here rather than in the config: it trades rewrite cost against how
# much work an interrupt loses, and the parquet is a few hundred KB.
FLUSH_EVERY = 10


def render_answer_prompt(client: CachedClient, cfg: Config, question: Question) -> str:
    """The one prompt every signal is ultimately about."""
    user = cfg.prompts.user_template.format(question=question.question)
    return _apply_template(client, cfg.prompts.system, user)


def _apply_template(client: CachedClient, system: str, user: str) -> str:
    """Chat-template the message pair when the backend has a template.

    A local causal LM needs the template applied by hand; an OpenAI-shaped
    endpoint applies its own server-side and wants the bare text.
    """
    apply = getattr(client.inner, "apply_chat_template", None)
    if callable(apply):
        return str(apply(system, user))
    return user if not system else f"{system}\n\n{user}"


def _verification_prompt(client: CachedClient, template: str, **fields: str) -> str:
    return _apply_template(client, "", template.format(**fields))


def _verify(
    client: CachedClient,
    prompt: str,
    tokens: TrueFalseTokens | None,
) -> VerificationResult | None:
    """One P(True) reading, or None when the backend cannot supply one.

    An OpenAI-shaped gateway that returns no logprobs cannot answer this
    question at all, and the honest representation of that is a missing value
    rather than a number derived from a text parse.
    """
    if tokens is None:
        return None
    return score_p_true(prompt, client.inner, tokens)  # type: ignore[arg-type]


def _verification_to_dict(result: VerificationResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "p_true": result.p_true,
        "true_token": result.true_token,
        "false_token": result.false_token,
        "raw_text": result.raw_text,
        "parse_failed": result.parse_failed,
    }


def verification_from_dict(raw: dict[str, Any] | None) -> VerificationResult | None:
    """Inverse of `_verification_to_dict`, used by the signal-scoring stage."""
    if raw is None:
        return None
    p_true = raw.get("p_true")
    return VerificationResult(
        p_true=None if p_true is None else float(p_true),
        true_token=raw.get("true_token"),
        false_token=raw.get("false_token"),
        raw_text=str(raw.get("raw_text", "")),
        parse_failed=bool(raw.get("parse_failed", False)),
    )


def generation_from_dict(raw: dict[str, Any]) -> Generation:
    """Re-expose the client's deserializer for downstream stages."""
    return _generation_from_dict(raw)


def _one_question(
    client: CachedClient,
    cfg: Config,
    question: Question,
    tokens: TrueFalseTokens | None,
) -> dict[str, Any]:
    """All generator work for a single question."""
    prompt = render_answer_prompt(client, cfg, question)

    greedy = client.generate(
        prompt,
        temperature=cfg.greedy.temperature,
        top_p=cfg.greedy.top_p,
        seed=cfg.greedy.seed,
        max_new_tokens=cfg.model_under_test.max_new_tokens,
        top_logprobs=cfg.greedy.top_logprobs,
        n=1,
    )[0]
    greedy_answer = clean_model_answer(greedy.text, abstain_token=cfg.prompts.abstain_token)

    # One call per seed. A single n=N call would share one seed across the whole
    # batch, so the "independent samples" would not be independently seeded and
    # the ablation over sample count would not be reproducible per sample.
    samples: list[Generation] = []
    for index in range(cfg.sampling.n_samples):
        samples.extend(
            client.generate(
                prompt,
                temperature=cfg.sampling.temperature,
                top_p=cfg.sampling.top_p,
                seed=cfg.sampling.seed_for(index),
                max_new_tokens=cfg.model_under_test.max_new_tokens,
                top_logprobs=0,
                n=1,
            )
        )
    sample_answers = [
        clean_model_answer(s.text, abstain_token=cfg.prompts.abstain_token) for s in samples
    ]

    verify_plain = _verify(
        client,
        _verification_prompt(
            client,
            cfg.prompts.verify_template,
            question=question.question,
            answer=greedy_answer or "(empty)",
        ),
        tokens,
    )
    verify_with_samples = _verify(
        client,
        _verification_prompt(
            client,
            cfg.prompts.verify_with_samples_template,
            question=question.question,
            answer=greedy_answer or "(empty)",
            samples="; ".join(a or "(empty)" for a in sample_answers),
        ),
        tokens,
    )

    confidence_prompt = _verification_prompt(
        client,
        cfg.prompts.verbal_confidence_template,
        question=question.question,
        answer=greedy_answer or "(empty)",
    )
    confidence_text = client.generate(
        confidence_prompt,
        temperature=cfg.greedy.temperature,
        top_p=cfg.greedy.top_p,
        seed=cfg.greedy.seed,
        max_new_tokens=cfg.model_under_test.max_new_tokens,
        top_logprobs=0,
        n=1,
    )[0].text
    confidence = parse_verbal_confidence(confidence_text.strip(), cfg.signals)

    return {
        "qid": question.qid,
        "dataset": question.dataset,
        "question": question.question,
        "gold_answers": json_str(list(question.gold_answers)),
        "greedy": json_str(_generation_to_dict(greedy)),
        "samples": json_str([_generation_to_dict(s) for s in samples]),
        "greedy_text": greedy.text,
        "greedy_answer": greedy_answer,
        "sample_answers": json_str(sample_answers),
        "verify_plain": json_str(_verification_to_dict(verify_plain)),
        "verify_with_samples": json_str(_verification_to_dict(verify_with_samples)),
        "verbal_confidence": float("nan") if confidence is None else float(confidence),
        "verbal_confidence_raw": confidence_text.strip(),
        "latency_s": greedy.latency_s + sum(s.latency_s for s in samples),
    }


def run(cfg: Config, *, limit: int | None = None) -> int:
    """Generate for every question not already in the checkpoint."""
    paths = StagePaths.of(cfg)
    dataset = read_checkpoint(paths.dataset)
    if dataset is None:
        raise FileNotFoundError(f"{paths.dataset} is absent; run build-dataset first")
    questions = frame_to_questions(dataset)

    previous = read_checkpoint(paths.generations)
    already = done_qids(previous)
    todo = [q for q in questions if q.qid not in already]
    if limit is not None:
        todo = todo[:limit]
    print(
        f"[generate] {len(questions)} questions, {len(already)} already done, {len(todo)} to go",
        flush=True,
    )
    if not todo:
        return len(already)

    cache = ResponseCache(cfg.paths.cache_dir)
    client = build_client(cfg.model_under_test, cache)

    # Resolved once per run: it is a property of the tokenizer, not of a
    # question, and it raises loudly if the two words are not single tokens.
    tokens: TrueFalseTokens | None = None
    if hasattr(client.inner, "token_ids_for"):
        tokens = resolve_true_false_tokens(client.inner)  # type: ignore[arg-type]
        print(f"[generate] P(True) tokens: {tokens}", flush=True)
    else:
        print("[generate] backend exposes no token scorer; family C will be NaN", flush=True)

    rows: list[dict[str, Any]] = (
        [] if previous is None else list(previous.to_dict(orient="records"))  # type: ignore[arg-type]
    )
    progress = Progress("generate", len(todo))
    errors = 0
    for question in todo:
        try:
            rows.append(_one_question(client, cfg, question, tokens))
        except Exception as exc:
            # A single bad question must not discard an hour of completed work.
            # The failure is counted and reported, and the question is simply
            # absent from the artifact rather than present with invented values.
            errors += 1
            print(f"[generate] {question.qid} failed: {exc}", flush=True)
        if len(rows) % FLUSH_EVERY == 0:
            write_checkpoint(pd.DataFrame(rows), paths.generations)
        progress.tick(question.qid)

    write_checkpoint(pd.DataFrame(rows), paths.generations)
    timing = progress.summary()
    timing["failures"] = float(errors)
    timing["cache_hits"] = float(cache.stats()["hits"])
    timing["cache_misses"] = float(cache.stats()["misses"])
    merge_timings(paths.timings, "generate", timing)
    print(
        f"[generate] wrote {len(rows)} rows to {paths.generations} "
        f"({timing['seconds_per_item']:.1f} s/question, {errors} failures)",
        flush=True,
    )

    # Drop the generator before returning so a caller that chains stages in one
    # process does not hold 1.57 GB while the NLI model tries to load.
    del client
    gc.collect()
    return len(rows)
