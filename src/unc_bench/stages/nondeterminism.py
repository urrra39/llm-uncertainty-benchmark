"""Probe: is "greedy" actually reproducible on this backend?

Every signal in family A is computed on one greedy generation, and the whole
project assumes that generation is a fixed function of the prompt. On CPU with
bfloat16 that assumption is worth checking rather than asserting: reduction order
in a matmul is not guaranteed stable across runs, and a single flipped token
changes the answer's logprob profile and therefore its signal values.

The probe bypasses the response cache deliberately. Going through the cache would
return the stored reply and report perfect determinism no matter what the model
does, which is the exact opposite of what is being measured.
"""

from __future__ import annotations

import json
from typing import Any

from unc_bench.client import LocalTransformersClient
from unc_bench.config import Config
from unc_bench.datasets.base import frame_to_questions
from unc_bench.normalize import normalize_answer
from unc_bench.stages.common import Progress, StagePaths, read_checkpoint


def run(cfg: Config) -> dict[str, Any]:
    """Regenerate N greedy answers twice and report disagreement."""
    paths = StagePaths.of(cfg)
    dataset = read_checkpoint(paths.dataset)
    if dataset is None:
        raise FileNotFoundError(f"{paths.dataset} is absent; run build-dataset first")
    questions = frame_to_questions(dataset)[: cfg.analysis.nondeterminism_probe_n]

    if cfg.model_under_test.backend != "local_transformers":
        raise NotImplementedError("the nondeterminism probe only supports the local backend")
    client = LocalTransformersClient(cfg.model_under_test)

    text_mismatches = 0
    answer_mismatches = 0
    logprob_deltas: list[float] = []
    progress = Progress("nondet", len(questions), every=10)
    for question in questions:
        user = cfg.prompts.user_template.format(question=question.question)
        prompt = client.apply_chat_template(cfg.prompts.system, user)
        first, second = (
            client.generate(
                prompt,
                temperature=cfg.greedy.temperature,
                top_p=cfg.greedy.top_p,
                seed=cfg.greedy.seed,
                max_new_tokens=cfg.model_under_test.max_new_tokens,
                top_logprobs=cfg.greedy.top_logprobs,
                n=1,
            )[0]
            for _ in range(2)
        )
        if first.text != second.text:
            text_mismatches += 1
        if normalize_answer(first.text) != normalize_answer(second.text):
            answer_mismatches += 1
        if first.answer_token_logprobs and second.answer_token_logprobs:
            logprob_deltas.append(
                abs(
                    first.answer_token_logprobs[0].logprob - second.answer_token_logprobs[0].logprob
                )
            )
        progress.tick(question.qid)

    n = len(questions)
    payload = {
        "n_probed": n,
        "text_mismatch_rate": text_mismatches / n if n else float("nan"),
        "normalized_answer_mismatch_rate": answer_mismatches / n if n else float("nan"),
        "max_first_token_logprob_delta": max(logprob_deltas) if logprob_deltas else 0.0,
    }
    out = cfg.paths.artifacts_dir / "nondeterminism.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("[nondeterminism] " + json.dumps(payload, sort_keys=True), flush=True)
    return payload
