"""Signal registry and the single place orientation is declared.

Every signal in this project is computed in its natural units. `mean_logprob` is
a logprob, `perplexity` is a perplexity, `p_true` is a probability. Those point
in opposite directions with respect to the thing being predicted, and the thing
being predicted is INCORRECT.

An inverted signal does not fail loudly. It reports AUROC 1-x, so a signal that
is actually strong at 0.72 reads as 0.28 and one that is useless at 0.50 reads
as 0.50 either way. That failure is invisible in a results table, which is why
orientation lives in exactly one place here rather than being applied ad hoc at
each call site.

The rule: `SignalSpec.orientation` says what the RAW value means. `oriented()`
flips it if needed so the exported number always means "higher = more likely
wrong". Nothing else in the codebase is allowed to negate a signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Raw value rises as the answer gets more likely to be WRONG. Exported as-is.
ORIENT_RISK = "risk"
# Raw value rises as the answer gets more likely to be RIGHT. Exported negated.
ORIENT_CONFIDENCE = "confidence"

FAMILY_A = "A"  # token logprobs of the greedy answer, 1x cost
FAMILY_B = "B"  # self-consistency over N samples, ~6x cost
FAMILY_C = "C"  # self-verification, 2x cost
FAMILY_T = "T"  # trivial baselines, 0x cost

FAMILY_LABELS: dict[str, str] = {
    FAMILY_A: "A. token logprobs",
    FAMILY_B: "B. self-consistency",
    FAMILY_C: "C. self-verification",
    FAMILY_T: "T. trivial baselines",
}


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """One scalar signal, its family, and what direction its raw value points."""

    name: str
    family: str
    orientation: str
    description: str

    def __post_init__(self) -> None:
        if self.orientation not in (ORIENT_RISK, ORIENT_CONFIDENCE):
            raise ValueError(f"{self.name}: unknown orientation {self.orientation!r}")
        if self.family not in FAMILY_LABELS:
            raise ValueError(f"{self.name}: unknown family {self.family!r}")

    def oriented(self, raw: float) -> float:
        """Convert a raw value to the exported convention: higher = more wrong.

        NaN passes through unchanged. NaN is the missing-value marker for this
        project (an empty answer has no mean logprob) and negating it would be
        meaningless, but it must not become 0.0 either, since 0.0 is a perfectly
        legitimate value for several of these signals.
        """
        value = float(raw)
        if math.isnan(value):
            return value
        return value if self.orientation == ORIENT_RISK else -value


_REGISTRY: dict[str, SignalSpec] = {}


def register(spec: SignalSpec) -> SignalSpec:
    """Add a signal to the global registry. Duplicate names are an error."""
    if spec.name in _REGISTRY:
        raise ValueError(f"signal {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> SignalSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no signal registered under {name!r}") from None


def registry() -> dict[str, SignalSpec]:
    """Read-only view of every registered signal."""
    return dict(_REGISTRY)


def signal_names(family: str | None = None) -> list[str]:
    """Registered signal names, in registration order, optionally one family."""
    return [name for name, spec in _REGISTRY.items() if family is None or spec.family == family]


def orient_all(raw_values: dict[str, float]) -> dict[str, float]:
    """Apply each signal's declared orientation to a whole row.

    Raises on an unregistered key. A signal that reaches the results table
    without a declared orientation is exactly the bug this module exists to
    prevent, so silently passing it through is not an option.
    """
    return {name: get_spec(name).oriented(value) for name, value in raw_values.items()}


def _clear_registry_for_tests() -> None:  # pragma: no cover - test helper
    _REGISTRY.clear()
