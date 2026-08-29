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
from typing import Any, Protocol, runtime_checkable

from unc_bench.cache import ResponseCache, cache_key
from unc_bench.config import ModelSpec
from unc_bench.types import Generation, TokenLogprob


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


class LocalTransformersClient:
    """HuggingFace causal LM on CPU with exact per-token logprobs.

    Only the generated tokens are scored. The prompt is never included in
    `answer_token_logprobs`, so no aggregate can accidentally average over
    instruction boilerplate.
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
        # 2 cores available; leaving this to the default picked 1 thread.
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self._tokenizer = AutoTokenizer.from_pretrained(spec.name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[spec.dtype]
        # `torch_dtype`, not `dtype`. transformers 4.46 is pinned here, and its
        # `from_pretrained` has no `dtype` parameter: passing one raises a
        # TypeError from the model constructor rather than being ignored. The
        # rename to `dtype` landed in a later release.
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self._model.eval()

    @property
    def backend(self) -> str:
        return "local_transformers"

    @property
    def model_name(self) -> str:
        return self._spec.name

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def _encode(self, prompt: str) -> Any:
        return self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

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
        torch = self._torch
        enc = self._encode(prompt)
        prompt_len = int(enc["input_ids"].shape[1])
        do_sample = temperature > 0.0

        torch.manual_seed(seed)
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

        eos_id = self._tokenizer.eos_token_id
        results: list[Generation] = []
        for row in range(n):
            token_ids = out.sequences[row][prompt_len:].tolist()
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
            results.append(
                Generation(
                    text=text,
                    answer_token_logprobs=tuple(per_token),
                    is_greedy=not do_sample,
                    seed=seed,
                    temperature=0.0 if not do_sample else temperature,
                    prompt_tokens=prompt_len,
                    completion_tokens=len(per_token),
                    latency_s=elapsed / n,
                )
            )
        return results

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
