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
    payload: dict[str, Any] = {
        "n_probed": n,
        "text_mismatch_rate": text_mismatches / n if n else float("nan"),
        "normalized_answer_mismatch_rate": answer_mismatches / n if n else float("nan"),
        "max_first_token_logprob_delta": max(logprob_deltas) if logprob_deltas else 0.0,
    }
    payload["batch_invariance"] = batch_invariance(
        cfg, [q.question for q in questions[:BATCH_PROMPTS]]
    )
    out = cfg.paths.artifacts_dir / "nondeterminism.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("[nondeterminism] " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


#: Widths compared by the batch-invariance harness, and prompts used. Four
#: prompts bound the cost: every width runs one greedy and one sampled pass
#: per prompt group.
BATCH_WIDTHS: tuple[int, ...] = (1, 2, 4, 8)
BATCH_PROMPTS = 4


def batch_invariance(cfg: Config, question_texts: list[str]) -> dict[str, Any]:
    """Compare widths 1/2/4/8 on greedy text, logprobs and sampled continuations.

    D27 stays open until this passes on the run's actual target device and
    dtype (a T4 in fp16 for run #3): the CPU/bfloat16 bit-identity measurement
    does not transfer to CUDA, where batched GEMM selects kernels by batch
    dimension. Each category is reported separately because they fail
    independently — greedy logprobs (family A) can be invariant while sampled
    continuations (family B) move with per-forward reseeding, which is exactly
    what per-prompt seeds (C2) address. `pass` is true only when every category
    matches width 1 exactly; it is a measurement, not a claim, and carries the
    device and dtype it was measured on so a CPU pass can never close the GPU
    question.
    """
    from unc_bench.client import LocalTransformersClient

    per_width: dict[str, Any] = {}
    reference: dict[str, Any] | None = None
    for width in BATCH_WIDTHS:
        spec = cfg.model_under_test.model_copy(update={"generation_batch_size": width})
        client = LocalTransformersClient(spec)
        rendered = [_render(client, cfg, text) for text in question_texts[:BATCH_PROMPTS]]
        prompt_seeds = [5000 + i for i in range(len(rendered))]
        greedy = client.generate_batch(
            rendered,
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_new_tokens=cfg.model_under_test.max_new_tokens,
            top_logprobs=cfg.greedy.top_logprobs,
            n=1,
        )
        sampled = client.generate_batch(
            rendered,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            seed=0,
            max_new_tokens=cfg.model_under_test.max_new_tokens,
            top_logprobs=0,
            n=1,
            seeds=prompt_seeds,
        )
        entry: dict[str, Any] = {
            "greedy_texts": [g[0].text for g in greedy],
            "greedy_logprobs": [[t.logprob for t in g[0].answer_token_logprobs] for g in greedy],
            "sampled_texts": [g[0].text for g in sampled],
        }
        if reference is None:
            reference = entry
            entry["matches_width_1"] = True
            entry["max_abs_logprob_delta_vs_width_1"] = 0.0
        else:
            assert reference is not None
            ref_texts: list[str] = reference["greedy_texts"]
            ref_sampled: list[str] = reference["sampled_texts"]
            ref_probs: list[list[float]] = reference["greedy_logprobs"]
            got_texts: list[str] = entry["greedy_texts"]
            got_sampled: list[str] = entry["sampled_texts"]
            got_probs: list[list[float]] = entry["greedy_logprobs"]
            texts_match = got_texts == ref_texts
            sampled_match = got_sampled == ref_sampled
            delta = 0.0
            for mine, base in zip(got_probs, ref_probs, strict=True):
                if len(mine) != len(base):
                    delta = float("inf")
                    break
                delta = max(delta, *(abs(a - b) for a, b in zip(mine, base, strict=True)))
            entry["matches_width_1"] = bool(texts_match and sampled_match and delta == 0.0)
            entry["greedy_texts_match_width_1"] = texts_match
            entry["sampled_texts_match_width_1"] = sampled_match
            entry["max_abs_logprob_delta_vs_width_1"] = None if delta == float("inf") else delta
        per_width[str(width)] = entry
    probe = LocalTransformersClient(cfg.model_under_test)
    passed = all(entry.get("matches_width_1", False) for w, entry in per_width.items() if w != "1")
    return {
        "widths": list(BATCH_WIDTHS),
        "device": probe.device,
        "dtype": cfg.model_under_test.dtype,
        "model": cfg.model_under_test.name,
        "pass": passed,
        "d27_closure_rule": (
            "D27 closes only when pass is true measured on the run's target "
            "device and dtype (T4, fp16 for run #3)"
        ),
        "per_width": per_width,
    }


def _render(client: LocalTransformersClient, cfg: Config, question_text: str) -> str:
    user = cfg.prompts.user_template.format(question=question_text)
    return client.apply_chat_template(cfg.prompts.system, user)
