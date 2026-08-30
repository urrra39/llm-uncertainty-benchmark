"""Device, dtype and batch-size resolution, tested without a GPU or weights.

`LocalTransformersClient.__init__` downloads a model, so the policy it applies is
extracted into `resolve_device` and `resolve_dtype` and tested against a stub
torch module. That is the whole point of pulling them out: CI installs neither
torch nor transformers, and the decision "CUDA if available, else CPU" is three
lines that must not be guessed at.

Every test here passes on a machine with no GPU, which is the machine this was
written on and the machine CI runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from unc_bench.client import resolve_device, resolve_dtype
from unc_bench.config import ModelSpec


def _torch_stub(*, cuda_available: bool) -> Any:
    """Minimal stand-in for the parts of torch these functions touch."""
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        float32="STUB_F32",
        bfloat16="STUB_BF16",
        float16="STUB_F16",
    )


def _spec(**overrides: Any) -> ModelSpec:
    base: dict[str, Any] = {
        "backend": "local_transformers",
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "dtype": "float16",
    }
    base.update(overrides)
    return ModelSpec(**base)


# --------------------------------------------------------------------------
# device: auto
# --------------------------------------------------------------------------


def test_auto_picks_cuda_when_available() -> None:
    assert resolve_device(_spec(device="auto"), _torch_stub(cuda_available=True)) == "cuda"


def test_auto_falls_back_to_cpu_without_cuda() -> None:
    """The case that matters here: no GPU must not be an error."""
    assert resolve_device(_spec(device="auto"), _torch_stub(cuda_available=False)) == "cpu"


def test_auto_is_the_default() -> None:
    # A config that says nothing about hardware gets detection, so run #2's
    # config runs on CPU here and on a GPU elsewhere with no edit.
    assert _spec().device == "auto"
    assert resolve_device(_spec(), _torch_stub(cuda_available=False)) == "cpu"


# --------------------------------------------------------------------------
# device: pinned
# --------------------------------------------------------------------------


def test_cpu_is_honoured_even_when_cuda_is_available() -> None:
    """This is what makes the CPU path testable on a GPU box.

    Without an override, "does run #2's configuration still work" could only be
    asked on hardware where CPU is the only option, which is the hardware least
    likely to be running the check.
    """
    assert resolve_device(_spec(device="cpu"), _torch_stub(cuda_available=True)) == "cpu"


def test_explicit_cuda_raises_when_unavailable() -> None:
    """Silently falling back would produce a run at a fraction of the expected
    speed with nothing in the output to explain it."""
    with pytest.raises(RuntimeError, match="device: cuda"):
        resolve_device(_spec(device="cuda"), _torch_stub(cuda_available=False))


def test_explicit_cuda_is_returned_when_available() -> None:
    assert resolve_device(_spec(device="cuda"), _torch_stub(cuda_available=True)) == "cuda"


# --------------------------------------------------------------------------
# dtype
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("float32", "STUB_F32"), ("bfloat16", "STUB_BF16"), ("float16", "STUB_F16")],
)
def test_dtype_names_map_to_torch_dtypes(name: str, expected: str) -> None:
    assert resolve_dtype(_spec(dtype=name), _torch_stub(cuda_available=False)) == expected


def test_run2_dtype_still_resolves() -> None:
    """Run #2 ran bfloat16 on CPU. That combination must keep working."""
    spec = _spec(name="Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16", device="auto")
    torch = _torch_stub(cuda_available=False)
    assert resolve_device(spec, torch) == "cpu"
    assert resolve_dtype(spec, torch) == "STUB_BF16"
    assert spec.generation_batch_size == 1


# --------------------------------------------------------------------------
# batch size, and the guards on it
# --------------------------------------------------------------------------


def test_batch_size_defaults_to_one() -> None:
    """Batch size 1 is one prompt per forward pass, i.e. run #2's call pattern.

    The default has to be 1 rather than something larger, because a config
    written before batching existed must not silently change behaviour.
    """
    assert _spec().generation_batch_size == 1


def test_batch_size_comes_from_config_not_a_constant() -> None:
    assert _spec(generation_batch_size=8).generation_batch_size == 8


def test_batch_size_is_bounded() -> None:
    with pytest.raises(ValueError):
        _spec(generation_batch_size=0)
    with pytest.raises(ValueError):
        _spec(generation_batch_size=1000)


def test_batch_size_rejected_on_the_api_backend() -> None:
    """A client-side batch size on an OpenAI-shaped endpoint does nothing.

    A number in a config that has no effect is worse than a missing one: it reads
    as a setting that was applied.
    """
    with pytest.raises(ValueError, match="local_transformers"):
        ModelSpec(backend="openai_compatible", name="gpt-5-mini", generation_batch_size=8)


def test_device_rejected_on_the_api_backend() -> None:
    with pytest.raises(ValueError, match="local_transformers"):
        ModelSpec(backend="openai_compatible", name="gpt-5-mini", device="cuda")


def test_api_backend_accepts_the_defaults() -> None:
    # The judges are openai_compatible and must keep validating unchanged.
    spec = ModelSpec(backend="openai_compatible", name="gpt-5-mini", max_new_tokens=8)
    assert spec.generation_batch_size == 1
    assert spec.device == "auto"
