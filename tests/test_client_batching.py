"""Ragged batching is closed, tested without torch, weights or a GPU.

D27 (docs/DECISIONS.md, session 7) measured the property the fix is built on:
a uniform-length batch is bit-identical to batch size 1, a padded ragged batch
is not. The fix is therefore to never build a padded batch — `generate_batch`
splits each chunk into equal-length forward passes — and what needs testing is
exactly that: what the model is handed, not what it computes.

These tests drive `LocalTransformersClient.generate_batch` against a stub
tokenizer, stub model and stub torch, the same way `resolve_device` is tested
in `test_client_device.py`. The stub tokenizer pads for real (left, to the
longest prompt) when asked, so a bucketing regression shows up as a pad token
inside a forward pass and fails the assertions below. The stub model records
every forward pass it sees and answers deterministically, with the generated
token ids encoding the output row index so that prompt-major result mapping is
assertable rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, overload

import pytest

from unc_bench.client import LocalTransformersClient, bucket_indices_by_length
from unc_bench.config import ModelSpec

PAD_ID = 0
BOS_ID = 1
EOS_ID = 200
VOCAB = 256


class _Vec:
    """A read-only vector: the slice of torch these tests touch.

    Holds ints for id rows and floats for score rows, like the real tensors.
    Values are stored as given — coercing ids to float breaks indexing.
    """

    def __init__(self, values: Sequence[float]) -> None:
        self._values: list[Any] = list(values)

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> _Vec: ...
    def __getitem__(self, index: int | slice) -> Any:
        selected = self._values[index]
        if isinstance(selected, list):
            return _Vec(selected)
        return selected

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def tolist(self) -> list[Any]:
        return list(self._values)

    def float(self) -> _Vec:
        return self

    def sum(self) -> Any:
        return sum(self._values)


class _Rows:
    """A 2D id/mask block with the `.shape` and row access the client uses."""

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self.shape = (len(rows), max((len(row) for row in rows), default=0))

    def __getitem__(self, index: int) -> _Vec:
        return _Vec(self._rows[index])

    def __iter__(self) -> Iterator[_Vec]:
        for row in self._rows:
            yield _Vec(row)

    def tolist(self) -> list[list[int]]:
        return [list(row) for row in self._rows]


class _Tokenizer:
    """One token per character behind a BOS prefix, so prompt length is
    `1 + len(prompt)` and two prompts share a length exactly when the bucket
    policy should group them. `padding=True` pads for real — left, to the
    longest row — exactly as the tokenizer behind the client would."""

    pad_token_id = PAD_ID
    eos_token_id = EOS_ID
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def __call__(
        self,
        prompts: list[str],
        *,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
        padding: bool = False,
    ) -> dict[str, Any]:
        del add_special_tokens  # the stub has no special tokens to add or skip
        sequences = [[BOS_ID] + [10 + ord(ch) % 40 for ch in prompt] for prompt in prompts]
        if return_tensors is None:
            return {"input_ids": sequences}
        if not padding:
            masks = [[1] * len(seq) for seq in sequences]
            return {"input_ids": _Rows(sequences), "attention_mask": _Rows(masks)}
        width = max(len(s) for s in sequences)
        ids: list[list[int]] = []
        mask: list[list[int]] = []
        for seq in sequences:
            pad = [PAD_ID] * (width - len(seq))
            if self.padding_side == "left":
                ids.append(pad + seq)
                mask.append([0] * len(pad) + [1] * len(seq))
            else:
                ids.append(seq + pad)
                mask.append([1] * len(seq) + [0] * len(pad))
        return {"input_ids": _Rows(ids), "attention_mask": _Rows(mask)}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        kept = [i for i in ids if i >= 10] if skip_special_tokens else ids
        return "".join(f"t{i}" for i in kept)


class _Torch:
    """`manual_seed` records; the numeric ops are identity/topk over plain floats."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def no_grad(self) -> Any:
        return nullcontext()

    def log_softmax(self, x: _Vec, dim: int = -1) -> _Vec:
        del dim  # the stub already holds logprobs
        return x

    def topk(self, x: _Vec, k: int) -> tuple[_Vec, _Vec]:
        order = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:k]
        return _Vec([x[i] for i in order]), _Vec(order)


class _Model:
    """Records every forward pass verbatim and answers deterministically.

    The token generated at step `s` by output row `r` is `55 + 10*r + s`, so a
    result's text names the row it came from and prompt-major mapping
    (`row = index * n + repeat`) is assertable. Scores put -0.1 on that token
    and -2.0 everywhere else, so top-k always contains the chosen token.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        input_ids: _Rows,
        attention_mask: _Rows,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
        num_return_sequences: int,
        return_dict_in_generate: bool,
        output_scores: bool,
        pad_token_id: int,
    ) -> SimpleNamespace:
        del attention_mask, temperature, top_p, return_dict_in_generate, output_scores, pad_token_id
        rows = input_ids.tolist()
        self.calls.append(
            {
                "rows": rows,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
            }
        )
        sequences: list[list[int]] = []
        for row_ids in rows:
            for _repeat in range(num_return_sequences):
                out_row = len(sequences)
                generated = [55 + 10 * out_row + step for step in range(max_new_tokens)]
                sequences.append(row_ids + generated)
        scores = [
            [
                _Vec([-0.1 if i == 55 + 10 * row + step else -2.0 for i in range(VOCAB)])
                for row in range(len(sequences))
            ]
            for step in range(max_new_tokens)
        ]
        return SimpleNamespace(
            sequences=[_Vec(row) for row in sequences],
            scores=scores,
        )


def _client(batch_size: int) -> tuple[LocalTransformersClient, _Model, _Torch]:
    """A client whose heavy internals are stubs, built past `__init__`."""
    spec = ModelSpec(
        backend="local_transformers",
        name="stub/stub-model",
        dtype="float16",
        device="cpu",
        generation_batch_size=batch_size,
    )
    client = LocalTransformersClient.__new__(LocalTransformersClient)
    client._spec = spec
    client._device = "cpu"
    torch = _Torch()
    client._torch = torch  # the real field is the module
    client._tokenizer = _Tokenizer()
    model = _Model()
    client._model = model
    return client, model, torch


def _greedy(client: LocalTransformersClient, prompts: list[str], *, n: int = 1) -> list[list[Any]]:
    return client.generate_batch(
        prompts,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=2,
        top_logprobs=5,
        n=n,
    )


# --------------------------------------------------------------------------
# the property D27 was about: the model never sees a padded generation batch
# --------------------------------------------------------------------------


def test_ragged_chunk_is_split_into_equal_length_forwards() -> None:
    # Lengths 3 / 5 / 3 / 7 tokens. The old code padded all four into one
    # forward pass; the fix gives three passes and no pass mixes lengths.
    client, model, _ = _client(batch_size=4)
    _greedy(client, ["ab", "abcd", "ab", "abcdef"])

    assert len(model.calls) == 3
    for call in model.calls:
        row_lengths = {len(row) for row in call["rows"]}
        assert len(row_lengths) == 1, f"a forward pass mixes lengths: {row_lengths}"
        assert all(PAD_ID not in row for row in call["rows"]), "a pad token reached the model"


def test_first_forward_holds_the_equal_length_prompts_in_arrival_order() -> None:
    client, model, _ = _client(batch_size=4)
    _greedy(client, ["ab", "abcd", "ab", "abcdef"])

    expected = [[BOS_ID, 10 + ord("a") % 40, 10 + ord("b") % 40]]
    assert model.calls[0]["rows"] == expected + expected


def test_equal_length_chunk_is_still_one_forward_pass() -> None:
    # Bucketing must not shatter a chunk the old code ran as one pass: four
    # same-length prompts at width 4 is exactly one forward of four rows.
    client, model, _ = _client(batch_size=4)
    _greedy(client, ["aa", "bb", "cc", "dd"])

    assert len(model.calls) == 1
    assert len(model.calls[0]["rows"]) == 4
    assert all(PAD_ID not in row for row in model.calls[0]["rows"])


def test_chunk_cap_remains_a_per_forward_pass_cap() -> None:
    # Five same-length prompts at width 2: outer chunking still applies first,
    # so the model sees forwards of 2, 2 and 1 rows — never a 5-row pass.
    client, model, _ = _client(batch_size=2)
    _greedy(client, ["aa", "bb", "cc", "dd", "ee"])

    assert [len(call["rows"]) for call in model.calls] == [2, 2, 1]


def test_seed_is_set_before_every_forward_pass() -> None:
    # A pass's output must be a function of the seed and its own inputs, not of
    # how many passes the chunk was split into.
    client, _, torch = _client(batch_size=4)
    _greedy(client, ["ab", "abcd", "ab", "abcdef"])

    assert torch.seeds == [0, 0, 0]


# --------------------------------------------------------------------------
# result contract: arrival order, prompt-major repeats, true prompt lengths
# --------------------------------------------------------------------------


def test_results_keep_arrival_order_with_n_greater_than_one() -> None:
    # Equal-length prompts, n=3, one forward pass. Each result names the
    # output row it came from: prompt i, repeat r must be row i*n+r,
    # prompt-major, within that forward pass.
    client, _, _ = _client(batch_size=3)
    prompts = ["ab", "cd", "ef"]
    results = _greedy(client, prompts, n=3)

    assert [len(per_prompt) for per_prompt in results] == [3, 3, 3]
    for i, per_prompt in enumerate(results):
        for r, generation in enumerate(per_prompt):
            expected_row = i * 3 + r
            assert generation.text.startswith(
                f"t{55 + 10 * expected_row}"
            ), f"prompt {i} repeat {r} carries the text of another row"
            assert generation.is_greedy is True
            assert generation.seed == 0
            assert generation.temperature == 0.0


def test_ragged_results_stay_in_arrival_order_per_forward_pass() -> None:
    # Ragged prompts are split into one forward pass per length bucket, and
    # each forward numbers its own rows from 0 — so row ids necessarily
    # collide across passes (both passes have a row 0). What must hold is
    # arrival order and prompt-major repeats *within* each pass.
    # Lengths here are [3, 5, 3]: bucket [0, 2] goes first (rows 0..5),
    # bucket [1] second (rows 0..2 of its own pass).
    client, model, _ = _client(batch_size=3)
    prompts = ["ab", "cdef", "gh"]
    results = _greedy(client, prompts, n=3)

    assert [len(per_prompt) for per_prompt in results] == [3, 3, 3]
    assert len(model.calls) == 2
    # First pass holds the two length-3 prompts; second the length-5 one.
    assert [len(call["rows"]) for call in model.calls] == [2, 1]
    for r, generation in enumerate(results[0]):
        assert generation.text.startswith(f"t{55 + 10 * r}")
    for r, generation in enumerate(results[2]):
        assert generation.text.startswith(f"t{55 + 10 * (3 + r)}")
    for r, generation in enumerate(results[1]):
        assert generation.text.startswith(f"t{55 + 10 * r}")
    for per_prompt in results:
        for generation in per_prompt:
            assert generation.is_greedy is True
            assert generation.seed == 0


def test_prompt_tokens_reports_true_lengths_not_padded_width() -> None:
    client, _, _ = _client(batch_size=4)
    prompts = ["ab", "abcd", "ab", "abcdef"]
    results = _greedy(client, prompts)

    assert [results[i][0].prompt_tokens for i in range(4)] == [3, 5, 3, 7]


def test_sampled_call_reports_its_temperature() -> None:
    client, model, _ = _client(batch_size=2)
    client.generate_batch(
        ["aa", "bb"],
        temperature=0.7,
        top_p=0.9,
        seed=1000,
        max_new_tokens=2,
        top_logprobs=0,
    )

    assert model.calls[0]["do_sample"] is True


def test_single_prompt_generate_delegates_to_the_batch_path() -> None:
    # `generate` is a delegation, not a second implementation. One prompt through
    # it must behave exactly like a one-prompt batch: one forward, one result.
    client, model, _ = _client(batch_size=4)
    out = client.generate(
        "ab",
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_new_tokens=2,
        top_logprobs=5,
    )

    assert len(model.calls) == 1
    assert len(out) == 1
    assert out[0].prompt_tokens == 3
    assert out[0].is_greedy is True


def test_empty_prompt_list_returns_empty() -> None:
    client, model, _ = _client(batch_size=4)
    assert (
        client.generate_batch(
            [], temperature=0.0, top_p=1.0, seed=0, max_new_tokens=2, top_logprobs=5
        )
        == []
    )
    assert model.calls == []


# --------------------------------------------------------------------------
# the pure grouping policy, specified directly
# --------------------------------------------------------------------------


def test_buckets_never_mix_lengths() -> None:
    lengths = [3, 5, 3, 7, 5, 3]
    for bucket in bucket_indices_by_length(lengths, cap=8):
        assert len({lengths[i] for i in bucket}) == 1


def test_buckets_partition_the_indices_exactly_once() -> None:
    lengths = [3, 5, 3, 7, 5, 3]
    buckets = bucket_indices_by_length(lengths, cap=2)
    flattened = [i for bucket in buckets for i in bucket]
    assert sorted(flattened) == list(range(len(lengths)))
    assert flattened == [0, 2, 5, 1, 4, 3]


def test_bucket_cap_splits_large_groups() -> None:
    buckets = bucket_indices_by_length([5] * 5, cap=2)
    assert buckets == [[0, 1], [2, 3], [4]]


def test_bucket_order_is_first_seen_and_stable_within() -> None:
    # Buckets come out in the order their length first appears, and indices
    # keep their input order inside a bucket, so a chunk of equal-length
    # prompts is one bucket in arrival order.
    assert bucket_indices_by_length([3, 5, 3, 7], cap=8) == [[0, 2], [1], [3]]
    assert bucket_indices_by_length([4, 4, 4], cap=4) == [[0, 1, 2]]


def test_bucket_empty_input() -> None:
    assert bucket_indices_by_length([], cap=4) == []


# --------------------------------------------------------------------------
# per-prompt seeds (C2): sampled tokens must not depend on batching
# --------------------------------------------------------------------------


def _sampled(
    client: LocalTransformersClient, prompts: list[str], seeds: list[int] | None, seed: int = 0
) -> list[list[Any]]:
    return client.generate_batch(
        prompts,
        temperature=0.7,
        top_p=0.9,
        seed=seed,
        max_new_tokens=2,
        top_logprobs=0,
        n=2,
        seeds=seeds,
    )


def test_sampled_outputs_match_across_widths_with_per_prompt_seeds() -> None:
    # Distinct seeds per prompt: every prompt gets a singleton forward seeded
    # individually, so widths 1 and 4 must produce identical texts per prompt.
    prompts = ["aa", "bb", "cc", "dd"]
    seeds = [101, 102, 103, 104]
    narrow, _, _ = _client(batch_size=1)
    wide, _, _ = _client(batch_size=4)
    narrow_out = _sampled(narrow, prompts, seeds)
    wide_out = _sampled(wide, prompts, seeds)

    assert [[g.text for g in per_prompt] for per_prompt in narrow_out] == [
        [g.text for g in per_prompt] for per_prompt in wide_out
    ]
    for per_prompt in wide_out:
        for generation in per_prompt:
            assert generation.is_greedy is False
            assert generation.temperature == 0.7


def test_distinct_seeds_split_a_bucket_into_singleton_forwards() -> None:
    client, model, torch = _client(batch_size=4)
    _sampled(client, ["aa", "bb", "cc", "dd"], [101, 102, 103, 104])

    assert [len(call["rows"]) for call in model.calls] == [1, 1, 1, 1]
    assert torch.seeds == [101, 102, 103, 104]


def test_shared_seed_keeps_the_single_forward_behaviour() -> None:
    # seeds=None is the historical path: one seed for every forward.
    client, model, torch = _client(batch_size=4)
    _sampled(client, ["aa", "bb", "cc", "dd"], None, seed=7)

    assert len(model.calls) == 1
    assert torch.seeds == [7]


def test_seed_count_mismatch_raises() -> None:
    client, _, _ = _client(batch_size=4)
    with pytest.raises(ValueError, match="one seed per prompt"):
        _sampled(client, ["aa", "bb"], [1])
