"""One interface, two backends.

`ModelClient` is the only thing the pipeline stages know about, so the model
under test is a config change rather than a code change. `LocalTransformersClient`
runs a HuggingFace causal LM on CPU and reads exact per-token logprobs off the
generation scores. `OpenAICompatibleClient` talks to any OpenAI-shaped endpoint,
including a vLLM server, and reads them off the `logprobs` field.

Both wrap every call in the response cache, so a rerun of a finished stage
issues no requests at all.

The heavy imports (`torch`, `transformers`) are deliberately deferred into the
local backend's constructor. CI installs neither, and every test drives a fake
client, so importing this module must stay cheap and dependency-free.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from unc_bench.cache import ResponseCache, cache_key
from unc_bench.config import ModelSpec
from unc_bench.types import Generation, TokenLogprob


def bucket_indices_by_length(lengths: Sequence[int], cap: int) -> list[list[int]]:
    """Group indices into same-length buckets of at most `cap` members.

    Pure and dependency-free so the grouping policy is testable without torch.
    The property everything else leans on is that a bucket never mixes lengths:
    D27 (docs/DECISIONS.md, session 7) measured that a uniform-length batch is
    bit-identical to batch size 1 while a padded ragged batch is not, so the
    fix for ragged batching is to never build one. Order is deterministic:
    buckets come out in first-seen order of their length, and indices keep
    their input order inside a bucket, so a chunk of equal-length prompts is
    still one bucket in arrival order.
    """
    grouped: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        grouped.setdefault(length, []).append(index)
    buckets: list[list[int]] = []
    width = max(cap, 1)
    for members in grouped.values():
        for start in range(0, len(members), width):
            buckets.append(members[start : start + width])
    return buckets


@runtime_checkable
class ModelClient(Protocol):
    """What the pipeline needs from a model."""

    @property
    def backend(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
        top_logprobs: int = 0,
        n: int = 1,
    ) -> list[Generation]:
        """Return `n` completions with per-token logprobs over the answer span."""
        ...

    def next_token_logprobs(self, prompt: str, *, seed: int = 0) -> dict[str, float]:
        """Logprobs of candidate next tokens, used for P(True).

        Implementations return whatever they can see: the full vocabulary for a
        local model, the top-k for an API. Callers must not assume a token is
        present just because it is a plausible continuation.
        """
        ...


def _generation_to_dict(gen: Generation) -> dict[str, Any]:
    return {
        "text": gen.text,
        "answer_token_logprobs": [t.to_dict() for t in gen.answer_token_logprobs],
        "is_greedy": gen.is_greedy,
        "seed": gen.seed,
        "temperature": gen.temperature,
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "latency_s": gen.latency_s,
    }


def _generation_from_dict(raw: dict[str, Any]) -> Generation:
    return Generation(
        text=str(raw["text"]),
        answer_token_logprobs=tuple(
            TokenLogprob.from_dict(t) for t in raw["answer_token_logprobs"]
        ),
        is_greedy=bool(raw["is_greedy"]),
        seed=int(raw["seed"]),
        temperature=float(raw["temperature"]),
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        latency_s=float(raw.get("latency_s", 0.0)),
    )


class CachedClient:
    """Wraps a raw backend with the on-disk response cache.

    Kept separate from the backends so the caching logic is written once and
    tested once, and so a test can point a fake backend at a temp directory.
    """

    def __init__(self, inner: ModelClient, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def backend(self) -> str:
        return self.inner.backend

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
        top_logprobs: int = 0,
        n: int = 1,
    ) -> list[Generation]:
        key = cache_key(
            backend=self.inner.backend,
            model=self.inner.model_name,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_logprobs=top_logprobs,
            n=n,
            kind="generate",
        )
        cached = self.cache.get(key)
        if cached is not None:
            return [_generation_from_dict(g) for g in cached["generations"]]
        out = self.inner.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_logprobs=top_logprobs,
            n=n,
        )
        self.cache.put(key, {"generations": [_generation_to_dict(g) for g in out]})
        return out

    def next_token_logprobs(self, prompt: str, *, seed: int = 0) -> dict[str, float]:
        key = cache_key(
            backend=self.inner.backend,
            model=self.inner.model_name,
            prompt=prompt,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            max_new_tokens=1,
            kind="next_token_logprobs",
        )
        cached = self.cache.get(key)
        if cached is not None:
            return {str(k): float(v) for k, v in cached["logprobs"].items()}
        out = self.inner.next_token_logprobs(prompt, seed=seed)
        self.cache.put(key, {"logprobs": out})
        return out


def resolve_device(spec: ModelSpec, torch: Any) -> str:
    """Which device a local model should run on.

    Separated from the client so it is testable without weights: the decision is
    three lines of policy and the thing that makes it hard to test is that
    `LocalTransformersClient.__init__` downloads a model.

    `auto` means CUDA when it is genuinely usable and CPU otherwise. "Genuinely
    usable" is `torch.cuda.is_available()`, which is False on a CPU-only build of
    torch as well as on a machine with no card, so one check covers both. An
    explicit `cuda` that cannot be satisfied raises rather than falling back:
    a config that asks for a GPU and silently gets CPU would produce a run at
    1/40th of the expected speed with nothing in the output to say why.
    """
    if spec.device == "cpu":
        return "cpu"
    available = bool(torch.cuda.is_available())
    if spec.device == "cuda":
        if not available:
            raise RuntimeError(
                "config requests device: cuda but torch.cuda.is_available() is False. "
                "Either no GPU is present or this is a CPU-only torch build. "
                "Set device: auto to fall back to CPU deliberately."
            )
        return "cuda"
    return "cuda" if available else "cpu"


def resolve_dtype(spec: ModelSpec, torch: Any) -> Any:
    """Map the config's dtype name onto a torch dtype.

    A plain lookup, kept as a function so the client has no dtype table inline
    and so the mapping is asserted by a test rather than by reading the code.
    """
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[spec.dtype]


class LocalTransformersClient:
    """HuggingFace causal LM with exact per-token logprobs, on CUDA or CPU.

    Only the generated tokens are scored. The prompt is never included in
    `answer_token_logprobs`, so no aggregate can accidentally average over
    instruction boilerplate. Under batching the guarantee is stronger still:
    ragged chunks are bucketed into equal-length forward passes, so the model is
    never handed a padded generation batch and what it scores is bit-identical
    to the unbatched pass (docs/DECISIONS.md, session 9).

    Device, dtype and batch size all come from `ModelSpec`. Nothing here reads
    the machine except through `resolve_device`, and nothing is a constant:
    run #2's configuration (CPU, bfloat16, batch size 1) produces the same call
    pattern it always did.
    """

    def __init__(self, spec: ModelSpec) -> None:
        # Deferred import: CI installs neither torch nor transformers.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._spec = spec
        self._torch = torch
        self._device = resolve_device(spec, torch)
        # CPU thread pinning only. On CUDA the compute is not on these threads
        # and capping them throttles the host-side loader for no benefit. On CPU
        # this matters: leaving it to the default picked 1 thread of the 2
        # available, which is where run #2's throughput measurements come from.
        if self._device == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        self._tokenizer = AutoTokenizer.from_pretrained(spec.name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding for a decoder-only model. Right padding would place pad
        # tokens between the prompt and the first generated token, and the model
        # attends to them: the answer would be conditioned on padding. Set
        # unconditionally rather than only when batching, so the tokenizer's
        # behaviour does not depend on a batch-size setting.
        self._tokenizer.padding_side = "left"
        dtype = resolve_dtype(spec, torch)
        # `torch_dtype`, not `dtype`. transformers 4.46 is pinned here, and its
        # `from_pretrained` has no `dtype` parameter: passing one raises a
        # TypeError from the model constructor rather than being ignored. The
        # rename to `dtype` landed in a later release.
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self._model.to(self._device)
        self._model.eval()

    @property
    def backend(self) -> str:
        return "local_transformers"

    @property
    def model_name(self) -> str:
        return self._spec.name

    @property
    def device(self) -> str:
        """Where this client is actually running. Reported, not inferred."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Prompts per forward pass, from config."""
        return self._spec.generation_batch_size

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _to_device(self, encoded: Any) -> Any:
        """Move a tokenizer batch onto the model's device.

        A dict-like `BatchEncoding` rather than a tensor, so this iterates the
        values. Missing this is the classic CUDA failure: the model is on the
        GPU, the inputs are on the CPU, and torch raises
        "Expected all tensors to be on the same device" from inside `generate`.
        """
        if self._device == "cpu":
            return encoded
        return {k: v.to(self._device) for k, v in encoded.items()}

    def _encode(self, prompt: str) -> Any:
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return self._to_device(encoded)

    def _encode_batch(self, prompts: list[str]) -> Any:
        """Pad several prompts into one batch.

        Returns the batch plus each row's true (unpadded) prompt length, because
        with left padding the answer starts at the padded width for every row and
        the per-row prompt length is what `Generation.prompt_tokens` must report.
        """
        encoded = self._tokenizer(
            prompts, return_tensors="pt", add_special_tokens=False, padding=True
        )
        lengths = [int(m.sum()) for m in encoded["attention_mask"]]
        return self._to_device(encoded), lengths

    def _score_row(
        self,
        out: Any,
        row: int,
        *,
        generated_from: int,
        top_logprobs: int,
    ) -> tuple[str, tuple[TokenLogprob, ...]]:
        """Decode one output row and read its per-token logprobs.

        `generated_from` is where the answer starts in `out.sequences[row]`. With
        left padding that is the padded prompt width, identical for every row in
        the batch, which is exactly why left padding is used: a right-padded batch
        would need a different offset per row and the pad tokens would sit inside
        the answer span.
        """
        torch = self._torch
        eos_id = self._tokenizer.eos_token_id
        token_ids = out.sequences[row][generated_from:].tolist()
        per_token: list[TokenLogprob] = []
        for step, token_id in enumerate(token_ids):
            if token_id == eos_id:
                break  # do not score the stop token as part of the answer
            step_logits = out.scores[step][row].float()
            step_logprobs = torch.log_softmax(step_logits, dim=-1)
            chosen_lp = float(step_logprobs[token_id])
            top_k = max(top_logprobs, 1)
            top_vals, top_idx = torch.topk(step_logprobs, k=top_k)
            alternatives = {
                self._tokenizer.decode([int(i)]): float(v)
                for v, i in zip(top_vals.tolist(), top_idx.tolist(), strict=True)
            }
            token_text = self._tokenizer.decode([token_id])
            alternatives.setdefault(token_text, chosen_lp)
            per_token.append(
                TokenLogprob(
                    token=token_text,
                    logprob=chosen_lp,
                    top_logprobs=alternatives,
                )
            )
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, tuple(per_token)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
        top_logprobs: int = 0,
        n: int = 1,
    ) -> list[Generation]:
        """One prompt. Delegates to the batch path with a batch of one.

        Written as a delegation rather than as a separate implementation so there
        is one place where logprobs are read off `out.scores` and no possibility
        of the single and batched paths drifting apart.
        """
        return self.generate_batch(
            [prompt],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_new_tokens=max_new_tokens,
            top_logprobs=top_logprobs,
            n=n,
        )[0]

    def _true_lengths(self, prompts: list[str]) -> list[int]:
        """Tokenized prompt lengths, before any batching decision.

        One unpadded tokenizer pass over the chunk. The lengths are what
        `bucket_indices_by_length` groups on, so a ragged chunk is split before
        the model ever sees it rather than padded into a uniform shape.
        """
        encoded = self._tokenizer(prompts, add_special_tokens=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
        top_logprobs: int = 0,
        n: int = 1,
        seeds: Sequence[int] | None = None,
    ) -> list[list[Generation]]:
        """Several prompts, one result list per prompt, never a padded batch.

        D27 (docs/DECISIONS.md, session 7) measured that padding a ragged batch
        perturbs per-token logprobs by up to 2.52e-02 against the unbatched
        values — the values signal family A is computed from — while a
        uniform-length batch is bit-identical (0.00e+00) on the measured
        configuration (CPU, bfloat16). Each chunk is therefore split by
        `bucket_indices_by_length` into equal-length groups and each group gets
        its own forward pass: no pad token is ever inserted into a generation
        batch. The bit-identity half of that claim is measured on CPU/bfloat16
        only and is NOT established for CUDA/fp16, where batched GEMM selects
        kernels by batch dimension; see the reopened D27.

        `prompts` longer than `generation_batch_size` is split into consecutive
        chunks of that size first, so the cap remains a per-forward-pass cap.

        `seeds`, when given (one per prompt), makes sampled tokens independent
        of batching: prompts sharing a forward pass must share a seed, so each
        bucket is sub-split by seed and every sub-group gets its own forward
        seeded individually. With per-question seeds (C2) the sampled output is
        a function of the row and the index alone — identical across widths
        1/2/4, which `test_sampled_outputs_match_across_widths` pins. When
        `seeds` is None the single `seed` applies to every forward, exactly the
        historical behaviour. Greedy passes consume no randomness either way.

        Returned latency is the chunk's forwards' wall clock divided by the
        number of sequences in it. That is an amortized per-sequence figure, not
        a measured one, and it is what makes batching visible in the cost table:
        the same work over fewer forward passes reports a smaller per-sequence
        latency.
        """
        torch = self._torch
        if not prompts:
            return []
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(
                f"generate_batch needs one seed per prompt: got {len(seeds)} seeds "
                f"for {len(prompts)} prompts"
            )
        do_sample = temperature > 0.0
        width = max(1, self._spec.generation_batch_size)
        results: list[list[Generation]] = [[] for _ in prompts]

        for start in range(0, len(prompts), width):
            chunk = prompts[start : start + width]
            chunk_seeds = (
                [seed] * len(chunk) if seeds is None else list(seeds[start : start + width])
            )
            lengths = self._true_lengths(chunk)

            for bucket in bucket_indices_by_length(lengths, width):
                # Prompts sharing a forward must share a seed; sub-split the
                # bucket so a forward never mixes streams.
                by_seed: dict[int, list[int]] = {}
                for index in bucket:
                    by_seed.setdefault(chunk_seeds[index], []).append(index)
                for group_seed, members in by_seed.items():
                    self._forward(
                        torch,
                        chunk=chunk,
                        members=members,
                        start=start,
                        group_seed=group_seed,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        max_new_tokens=max_new_tokens,
                        top_logprobs=top_logprobs,
                        n=n,
                        results=results,
                    )

        if any(not filled for filled in results):
            # Unreachable if bucket_indices_by_length partitions correctly; kept
            # because the alternative on a bug is silently returning short rows.
            raise RuntimeError("internal batching error: a prompt never reached a forward pass")
        return results

    def _forward(
        self,
        torch: Any,
        *,
        chunk: list[str],
        members: list[int],
        start: int,
        group_seed: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        max_new_tokens: int,
        top_logprobs: int,
        n: int,
        results: list[list[Generation]],
    ) -> None:
        """One unpadded forward pass over a same-length, same-seed group."""
        bucket_prompts = [chunk[i] for i in members]
        enc, true_lengths = self._encode_batch(bucket_prompts)
        padded_len = int(enc["input_ids"].shape[1])

        torch.manual_seed(group_seed)
        started = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_return_sequences=n,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        per_sequence = elapsed / max(len(members) * n, 1)

        # `num_return_sequences=n` lays the output out prompt-major: rows
        # [0..n) belong to group prompt 0, [n..2n) to group prompt 1.
        # Indexed rather than assumed, because getting this wrong would
        # attribute one question's samples to another and nothing
        # downstream would notice.
        for position, index in enumerate(members):
            per_prompt: list[Generation] = []
            for repeat in range(n):
                row = position * n + repeat
                text, per_token = self._score_row(
                    out, row, generated_from=padded_len, top_logprobs=top_logprobs
                )
                per_prompt.append(
                    Generation(
                        text=text,
                        answer_token_logprobs=per_token,
                        is_greedy=not do_sample,
                        seed=group_seed,
                        temperature=0.0 if not do_sample else temperature,
                        prompt_tokens=true_lengths[position],
                        completion_tokens=len(per_token),
                        latency_s=per_sequence,
                    )
                )
            results[start + index] = per_prompt

    def next_token_logprobs(self, prompt: str, *, seed: int = 0) -> dict[str, float]:
        """Full-vocabulary next-token logprobs, keyed by decoded token text.

        Returns the whole vocabulary rather than a top-k, because P(True) needs
        the logprob of a specific token that may not be in any top-k.

        `seed` is accepted to satisfy the protocol and ignored: this is a single
        forward pass with no sampling, so it is already deterministic.
        """
        del seed
        torch = self._torch
        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self._model(**enc).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        return {
            self._tokenizer.convert_ids_to_tokens(int(i)): float(logprobs[i])
            for i in range(logprobs.shape[0])
        }

    def token_ids_for(self, text: str) -> list[int]:
        """Tokenizer probe used by the True-token assertion."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(i) for i in ids]

    def logprob_of_token_id(self, prompt: str, token_id: int) -> float:
        torch = self._torch
        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self._model(**enc).logits[0, -1].float()
        return float(torch.log_softmax(logits, dim=-1)[token_id])

    def logprobs_of_token_ids(self, prompt: str, token_ids: list[int]) -> list[float]:
        """Several token logprobs from ONE forward pass.

        P(True) needs two logprobs at the same position, and calling
        `logprob_of_token_id` twice runs the prompt through the model twice. That
        measured at 4.55 s for the pair on this machine against 2.3 s for one
        pass, and there are two verification variants per question, so the
        redundant pass costs about 4.6 s of the ~35 s question budget: 19 minutes
        over 250 questions for no information.
        """
        torch = self._torch
        enc = self._encode(prompt)
        with torch.no_grad():
            logits = self._model(**enc).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        return [float(logprobs[i]) for i in token_ids]

    def apply_chat_template(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rendered = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return str(rendered)


class OpenAICompatibleClient:
    """Any OpenAI-shaped endpoint, including a vLLM server.

    This is the path a GPU run would take. On the gateway available here the
    `logprobs` field comes back absent, which is why the local backend is the
    default; see docs/DECISIONS.md D1.
    """

    def __init__(self, spec: ModelSpec) -> None:
        # Deferred import to keep module import cheap.
        from openai import OpenAI

        self._spec = spec
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {spec.api_key_env} is unset")
        self._client = OpenAI(
            api_key=api_key,
            base_url=spec.base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=spec.request_timeout_s,
            max_retries=spec.max_retries,
        )

    @property
    def backend(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._spec.name

    def complete_text(
        self, prompt: str, *, seed: int = 0, max_new_tokens: int | None = None
    ) -> str:
        """Plain text completion. Used for the judges, which need no logprobs."""
        response = self._client.chat.completions.create(
            model=self._spec.name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_new_tokens or self._spec.max_new_tokens,
            seed=seed,
        )
        return response.choices[0].message.content or ""

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        seed: int,
        max_new_tokens: int,
        top_logprobs: int = 0,
        n: int = 1,
    ) -> list[Generation]:
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._spec.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_completion_tokens=max_new_tokens,
            n=n,
            logprobs=top_logprobs > 0,
            top_logprobs=top_logprobs if top_logprobs > 0 else None,
        )
        elapsed = time.perf_counter() - started
        results: list[Generation] = []
        for choice in response.choices:
            per_token: list[TokenLogprob] = []
            logprob_payload = getattr(choice, "logprobs", None)
            content = getattr(logprob_payload, "content", None) if logprob_payload else None
            for item in content or []:
                alternatives = {alt.token: float(alt.logprob) for alt in (item.top_logprobs or [])}
                alternatives.setdefault(item.token, float(item.logprob))
                per_token.append(
                    TokenLogprob(
                        token=item.token,
                        logprob=float(item.logprob),
                        top_logprobs=alternatives,
                    )
                )
            usage = response.usage
            results.append(
                Generation(
                    text=choice.message.content or "",
                    answer_token_logprobs=tuple(per_token),
                    is_greedy=temperature == 0.0,
                    seed=seed,
                    temperature=temperature,
                    prompt_tokens=int(usage.prompt_tokens) if usage else 0,
                    completion_tokens=int(usage.completion_tokens) if usage else 0,
                    latency_s=elapsed / max(len(response.choices), 1),
                )
            )
        return results

    def next_token_logprobs(self, prompt: str, *, seed: int = 0) -> dict[str, float]:
        """Top-k logprobs of the first generated token.

        An API can only report its top-k, so a token outside that set is
        genuinely unobservable here. Callers treat a missing token as such
        rather than substituting a floor value.
        """
        response = self._client.chat.completions.create(
            model=self._spec.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=1,
            seed=seed,
            logprobs=True,
            top_logprobs=20,
        )
        payload = getattr(response.choices[0], "logprobs", None)
        content = getattr(payload, "content", None) if payload else None
        if not content:
            return {}
        first = content[0]
        out = {alt.token: float(alt.logprob) for alt in (first.top_logprobs or [])}
        out.setdefault(first.token, float(first.logprob))
        return out


def build_client(spec: ModelSpec, cache: ResponseCache) -> CachedClient:
    """Construct the backend named in the config and wrap it in the cache."""
    inner: ModelClient
    if spec.backend == "local_transformers":
        inner = LocalTransformersClient(spec)
    elif spec.backend == "openai_compatible":
        inner = OpenAICompatibleClient(spec)
    else:  # pragma: no cover  pydantic already restricts the literal
        raise ValueError(f"unknown backend {spec.backend!r}")
    return CachedClient(inner, cache)
