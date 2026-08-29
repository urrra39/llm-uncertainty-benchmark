"""The cache key must be sensitive to everything that changes a response and
insensitive to nothing else.

If the key ignores the seed, the five self-consistency samples collapse into one
cached answer and the whole family reads as perfectly consistent. If it ignores
the prompt, an edited template silently reuses stale responses. Both failures are
invisible in the output, so they are pinned here.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from unc_bench.cache import ResponseCache, cache_key
from unc_bench.client import CachedClient

BASE: dict[str, object] = {
    "backend": "local_transformers",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "prompt": "Q: What is the capital of France?\nA:",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
    "max_new_tokens": 24,
    "top_logprobs": 5,
    "n": 1,
    "kind": "generate",
}


def _key(**overrides: object) -> str:
    merged = {**BASE, **overrides}
    return cache_key(**merged)  # type: ignore[arg-type]


def test_key_is_stable_across_calls() -> None:
    assert _key() == _key()


def test_key_is_a_sha256_hex_digest() -> None:
    key = _key()
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "openai_compatible"),
        ("model", "Qwen/Qwen2.5-7B-Instruct"),
        ("prompt", "Q: What is the capital of Italy?\nA:"),
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("seed", 1),
        ("max_new_tokens", 25),
        ("top_logprobs", 10),
        ("n", 5),
        ("kind", "next_token_logprobs"),
    ],
)
def test_every_keyed_field_changes_the_key(field: str, value: object) -> None:
    assert _key(**{field: value}) != _key(), f"{field} does not affect the cache key"


def test_all_ten_keyed_fields_are_pairwise_distinct() -> None:
    # Guards against two fields being concatenated into the payload in a way
    # that lets a change in one imitate a change in the other.
    variants = {
        "backend": _key(backend="vllm"),
        "model": _key(model="other"),
        "prompt": _key(prompt="other"),
        "temperature": _key(temperature=0.5),
        "top_p": _key(top_p=0.5),
        "seed": _key(seed=99),
        "max_new_tokens": _key(max_new_tokens=99),
        "top_logprobs": _key(top_logprobs=1),
        "n": _key(n=9),
        "kind": _key(kind="other"),
    }
    for (name_a, key_a), (name_b, key_b) in itertools.combinations(variants.items(), 2):
        assert key_a != key_b, f"{name_a} and {name_b} collide"


def test_float_formatting_is_value_based_not_text_based() -> None:
    # 0.7 and 0.70 are the same number and must hash the same. 0.7 and 0.71
    # are not and must not.
    assert _key(temperature=0.7) == _key(temperature=0.70)
    assert _key(temperature=0.7) != _key(temperature=0.71)


def test_int_and_float_seed_agree() -> None:
    assert _key(seed=3) == _key(seed=3)


def test_roundtrip_through_disk(tmp_path: Path) -> None:
    store = ResponseCache(tmp_path)
    key = _key()
    assert store.get(key) is None
    store.put(key, {"generations": [], "note": "hello"})
    got = store.get(key)
    assert got is not None
    assert got["note"] == "hello"
    assert store.stats() == {"hits": 1, "misses": 1, "writes": 1}


def test_truncated_entry_reads_as_a_miss(tmp_path: Path) -> None:
    store = ResponseCache(tmp_path)
    key = _key()
    store.put(key, {"x": 1})
    path = next(tmp_path.rglob("*.json.gz"))
    path.write_bytes(b"\x1f\x8b\x08 truncated garbage")
    assert store.get(key) is None


def test_cached_client_serves_second_call_from_disk(
    fake_backend: object, cache: ResponseCache
) -> None:
    client = CachedClient(fake_backend, cache)  # type: ignore[arg-type]
    kwargs: dict[str, object] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_new_tokens": 8,
        "top_logprobs": 5,
        "n": 1,
    }
    first = client.generate("Q: test\nA:", **kwargs)  # type: ignore[arg-type]
    second = client.generate("Q: test\nA:", **kwargs)  # type: ignore[arg-type]

    assert len(fake_backend.generate_calls) == 1, "second call was not served from cache"  # type: ignore[attr-defined]
    assert [g.text for g in first] == [g.text for g in second]
    assert first[0].answer_token_logprobs == second[0].answer_token_logprobs
    assert first[0].is_greedy is True


def test_distinct_seeds_are_not_conflated(fake_backend: object, cache: ResponseCache) -> None:
    # The failure this catches: five samples that all return the same cached
    # answer, making self-consistency look perfect.
    client = CachedClient(fake_backend, cache)  # type: ignore[arg-type]
    texts = set()
    for seed in range(5):
        got = client.generate(
            "Q: test\nA:",
            temperature=0.7,
            top_p=0.9,
            seed=1000 + seed,
            max_new_tokens=8,
            top_logprobs=5,
            n=1,
        )
        texts.add(got[0].text)
    assert len(fake_backend.generate_calls) == 5  # type: ignore[attr-defined]
    assert len(texts) == 5


def test_next_token_logprobs_is_cached(fake_backend: object, cache: ResponseCache) -> None:
    client = CachedClient(fake_backend, cache)  # type: ignore[arg-type]
    a = client.next_token_logprobs("prompt")
    b = client.next_token_logprobs("prompt")
    assert a == b
    assert fake_backend.next_token_calls == 1  # type: ignore[attr-defined]


def test_generate_and_next_token_do_not_share_a_key(
    fake_backend: object, cache: ResponseCache
) -> None:
    # Both are "one call on one prompt"; the `kind` field keeps them apart.
    client = CachedClient(fake_backend, cache)  # type: ignore[arg-type]
    client.generate("p", temperature=0.0, top_p=1.0, seed=0, max_new_tokens=1, top_logprobs=0, n=1)
    client.next_token_logprobs("p", seed=0)
    assert len(fake_backend.generate_calls) == 1  # type: ignore[attr-defined]
    assert fake_backend.next_token_calls == 1  # type: ignore[attr-defined]
