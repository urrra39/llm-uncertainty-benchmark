"""Shared fixtures. No test in this suite makes a live model call."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from unc_bench.cache import ResponseCache
from unc_bench.types import Generation, TokenLogprob


class FakeBackend:
    """Counts calls so cache behaviour is observable, and returns deterministic
    output derived from the prompt and seed."""

    def __init__(self, *, backend: str = "fake", model_name: str = "fake-model") -> None:
        self._backend = backend
        self._model_name = model_name
        self.generate_calls: list[dict[str, object]] = []
        self.next_token_calls: int = 0

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model_name(self) -> str:
        return self._model_name

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
        self.generate_calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "top_logprobs": top_logprobs,
                "n": n,
            }
        )
        out: list[Generation] = []
        for i in range(n):
            text = f"ans-{seed}-{i}"
            out.append(
                Generation(
                    text=text,
                    answer_token_logprobs=(
                        TokenLogprob("ans", -0.1, {"ans": -0.1, "other": -2.3}),
                        TokenLogprob(text[-1], -0.4, {text[-1]: -0.4, "x": -1.9}),
                    ),
                    is_greedy=temperature == 0.0,
                    seed=seed,
                    temperature=temperature,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=2,
                    latency_s=0.01,
                )
            )
        return out

    def next_token_logprobs(self, prompt: str, *, seed: int = 0) -> dict[str, float]:
        del prompt, seed  # deterministic stub; args exist to match the protocol
        self.next_token_calls += 1
        # Deterministic pseudo-distribution, normalized so the values are a
        # genuine log-probability vector.
        weights = {"True": 3.0, "False": 1.0, "Maybe": 0.5}
        total = sum(weights.values())
        return {k: math.log(v / total) for k, v in weights.items()}


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def cache(tmp_path: Path) -> ResponseCache:
    return ResponseCache(tmp_path / "cache")


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
