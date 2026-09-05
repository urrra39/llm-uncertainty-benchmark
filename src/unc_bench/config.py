"""Single source of truth for every tunable number.

Nothing in `src/` may hard-code a threshold, sample count, temperature, seed or
bin count. If a number affects a result it lives here and is validated on load,
so a config file that would silently produce a wrong analysis fails at startup
instead of halfway through a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Frozen(BaseModel):
    """Base: reject unknown keys so a typo in YAML is an error, not a no-op.

    `protected_namespaces` is cleared because several fields legitimately start
    with `model_` (`model_under_test`, `model_name`) and pydantic otherwise
    warns on every import.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ModelSpec(Frozen):
    """The model under test, or a judge."""

    backend: Literal["local_transformers", "openai_compatible"]
    name: str
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    max_new_tokens: int = Field(default=24, ge=1, le=512)
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)

    #: Where to run a `local_transformers` model. `auto` picks CUDA when
    #: `torch.cuda.is_available()` and CPU otherwise, which is what a config
    #: should say if it does not care. Naming a device explicitly overrides
    #: detection, and `cpu` is what makes the CPU path testable on a machine
    #: that has a GPU — without it, "does the CPU path still work" could only be
    #: asked on hardware where it is the only option.
    device: Literal["auto", "cuda", "cpu"] = "auto"

    #: How many prompts `LocalTransformersClient` puts through one forward pass.
    #: 1 is one prompt per call, which is exactly run #2's call pattern, so a
    #: config that omits this field behaves as it did before batching existed.
    #: Raising it is a GPU optimization: on 2 CPU cores a batch competes with
    #: itself for the same threads and buys nothing.
    #:
    #: The cap is per *forward pass*, not per question. The sampling loop in
    #: `stages/generate.py` still issues one call per seed, because a single
    #: `n=5` call would share one seed across the batch and the N-ablation needs
    #: each sample independently seeded (docs/DECISIONS.md D24).
    #:
    #: D27 (docs/DECISIONS.md, session 7) measured that a padded ragged batch
    #: perturbs per-token logprobs by up to 2.52e-02, so the cap used to be safe
    #: only at 1. Session 9 closed it: `generate_batch` buckets every chunk into
    #: equal-length forward passes, so any value is bit-identical to 1 for
    #: signal family A and a raised cap trades only memory and occupancy.
    generation_batch_size: int = Field(default=1, ge=1, le=256)

    @model_validator(mode="after")
    def _batching_is_local_only(self) -> ModelSpec:
        # An OpenAI-shaped endpoint batches server-side or not at all; a
        # client-side batch size would be a number that silently does nothing.
        if self.backend != "local_transformers" and self.generation_batch_size != 1:
            raise ValueError(
                "generation_batch_size applies to the local_transformers backend only; "
                f"backend is {self.backend!r}"
            )
        if self.backend != "local_transformers" and self.device != "auto":
            raise ValueError(
                f"device applies to the local_transformers backend only; "
                f"backend is {self.backend!r}"
            )
        return self


class GreedySpec(Frozen):
    """The one generation every signal is computed on."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 0
    top_logprobs: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _must_be_greedy(self) -> GreedySpec:
        # Trap: computing signal family A on a sampled generation. The greedy
        # spec is the anchor for everything, so refuse anything but temp 0.
        if self.temperature != 0.0:
            raise ValueError("greedy spec must have temperature == 0.0")
        return self


class SamplingSpec(Frozen):
    """The N extra samples used by the self-consistency family."""

    n_samples: int = Field(default=5, ge=1, le=32)
    # No `gt=0` here: the field constraint would pre-empt the validator below
    # and report a generic "greater_than" error instead of the real reason.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    seed_base: int = 1000
    ablation_n: tuple[int, ...] = (1, 2, 3, 5)

    @model_validator(mode="after")
    def _sane(self) -> SamplingSpec:
        if self.temperature == 0.0:
            raise ValueError("sampling temperature must be > 0, else samples are all identical")
        if max(self.ablation_n) > self.n_samples:
            raise ValueError(
                f"ablation_n max {max(self.ablation_n)} exceeds n_samples {self.n_samples}"
            )
        return self

    def seed_for(self, sample_index: int) -> int:
        """Distinct, reproducible seed per sample."""
        return self.seed_base + sample_index


class NLISpec(Frozen):
    """Bidirectional-entailment clustering for semantic entropy."""

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli"
    entailment_threshold: Probability = 0.5
    max_length: int = Field(default=128, ge=16, le=512)
    batch_size: int = Field(default=16, ge=1, le=256)
    # Set from config.id2label at load time; asserted, never assumed.
    entailment_label: str = "entailment"


class SignalsSpec(Frozen):
    """Numbers the signal functions need. Kept out of the signal code itself so
    no threshold is hard-coded in `src/unc_bench/signals/`."""

    # Wu et al. 2016 length penalty exponent for the length-normalized total.
    length_penalty_alpha: float = Field(default=0.6, ge=0.0, le=2.0)
    # Ceiling on the exponent inside perplexity's exp(). exp(20) is ~4.9e8:
    # large enough to rank a pathological row last, small enough that it cannot
    # overflow to inf and poison every bootstrap resample that draws the row.
    perplexity_clamp: float = Field(default=20.0, gt=0.0, le=100.0)
    # Ceiling on a verbalized confidence integer before it is rejected.
    verbal_confidence_max: int = Field(default=100, ge=1)


class DatasetMix(Frozen):
    """How many questions to draw from each source."""

    popqa: int = Field(ge=0)
    triviaqa: int = Field(ge=0)
    simpleqa: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.popqa + self.triviaqa + self.simpleqa

    @model_validator(mode="after")
    def _nonempty(self) -> DatasetMix:
        if self.total == 0:
            raise ValueError("dataset mix is empty")
        return self


class DifficultySpec(Frozen):
    """Knobs that set how hard the question set is.

    Run #1 drew unrestricted TriviaQA and got 7 correct answers out of 75
    answered, which made every AUROC in the table noise. These knobs exist to
    move the base rate into a range where discrimination is measurable, and they
    default to off so run #1's distribution is still reproducible.
    """

    #: Keep only TriviaQA questions whose answer entity looks well-known.
    triviaqa_easy_only: bool = False
    #: Keep only PopQA rows at or above this quantile of subject pageviews.
    #: None disables the filter. 0.9 is the top popularity decile.
    popqa_popularity_quantile: float | None = Field(default=None, ge=0.0, lt=1.0)
    #: Alias-count threshold for the TriviaQA easy slice. Raising it demands a
    #: more heavily redirected answer entity, i.e. a more famous one.
    triviaqa_min_aliases: int = Field(default=12, ge=1)
    #: Keep only these PopQA relation types. None keeps all 16. Pilot iteration
    #: 1 showed relation type dominates subject popularity as a difficulty axis:
    #: `capital` is a lookup, `screenwriter` is not.
    popqa_relations: tuple[str, ...] | None = None


class FewShotSpec(Frozen):
    """Fixed exemplar Q/A pairs prefixed to every generation prompt.

    Purpose is format stabilization, not teaching: a 0.5B model asked a bare
    question will sometimes answer in a sentence, which costs it an exact match
    it deserved and pollutes the answer-length signal. Two exemplars are enough
    to fix the shape and cheap enough not to matter at 24 output tokens.

    The exemplars are hard-coded in the config and asserted absent from the
    evaluation set (`assert_disjoint_from`), because an exemplar that leaked into
    the eval rows would be a memorization result presented as a knowledge result.
    """

    enabled: bool = False
    pairs: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _pairs_present_when_enabled(self) -> FewShotSpec:
        if self.enabled and len(self.pairs) < 1:
            raise ValueError("few_shot.enabled is true but no exemplar pairs were given")
        for question, answer in self.pairs:
            if not question.strip() or not answer.strip():
                raise ValueError("few_shot exemplars must have non-empty question and answer")
        return self

    def assert_disjoint_from(self, questions: Sequence[str]) -> None:
        """Fail loudly if any exemplar question is also an evaluation question.

        Normalized comparison, so a difference in punctuation or case does not
        let a duplicate through. Raises rather than warning: a contaminated run
        should not produce a results file at all.
        """
        if not self.enabled:
            return
        from unc_bench.normalize import normalize_answer

        evaluation = {normalize_answer(q) for q in questions}
        for question, _ in self.pairs:
            if normalize_answer(question) in evaluation:
                raise ValueError(f"few-shot exemplar leaked into the evaluation set: {question!r}")


class PilotGateSpec(Frozen):
    """The autonomous decision rule that replaces human approval."""

    n_questions: int = Field(default=100, ge=20)
    error_rate_low: Probability = 0.25
    error_rate_high: Probability = 0.65
    max_iterations: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def _ordered(self) -> PilotGateSpec:
        if self.error_rate_low >= self.error_rate_high:
            raise ValueError("error_rate_low must be < error_rate_high")
        return self


class JudgeSpec(Frozen):
    """Stage-2 and stage-3 label judges. Must differ from the model under test."""

    primary: ModelSpec
    secondary: ModelSpec
    cross_validation_n: int = Field(default=100, ge=20)
    kappa_trust_threshold: Probability = 0.7
    secondary_seed: int = 4242


class SplitSpec(Frozen):
    """Train split exists only to fit recalibrators. Never evaluated on."""

    train_fraction: Probability = 0.3
    seed: int = 20260101

    @model_validator(mode="after")
    def _leaves_room(self) -> SplitSpec:
        if not 0.05 <= self.train_fraction <= 0.6:
            raise ValueError("train_fraction outside 0.05..0.6 is not a useful split")
        return self


class AnalysisSpec(Frozen):
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    bootstrap_seed: int = 987654321
    ci_level: Probability = 0.95
    ece_bins: int = Field(default=15, ge=2, le=100)
    target_accuracy: Probability = 0.90
    logreg_cv_folds: int = Field(default=5, ge=2, le=20)
    logreg_seed: int = 31337
    # A gap this small is noise at n~1200; refuse to name a winner under it.
    min_meaningful_auroc_gap: float = Field(default=0.02, ge=0.0, le=0.5)
    holm_alpha: Probability = 0.05
    nondeterminism_probe_n: int = Field(default=50, ge=10)


class PathsSpec(Frozen):
    raw_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/cache")
    artifacts_dir: Path = Path("data/artifacts")
    figures_dir: Path = Path("figures")
    results_json: Path = Path("results.json")
    human_validation_csv: Path = Path("data/human_validation_sample.csv")


class PromptSpec(Frozen):
    """Frozen zero-shot prompt. Changing this invalidates the response cache,
    which is correct: the prompt is part of the cache key."""

    system: str
    user_template: str
    abstain_token: str = "UNKNOWN"
    verify_template: str
    verify_with_samples_template: str
    verbal_confidence_template: str

    @model_validator(mode="after")
    def _has_placeholders(self) -> PromptSpec:
        if "{question}" not in self.user_template:
            raise ValueError("user_template must contain {question}")
        for field_name in ("verify_template", "verify_with_samples_template"):
            text = getattr(self, field_name)
            for placeholder in ("{question}", "{answer}"):
                if placeholder not in text:
                    raise ValueError(f"{field_name} must contain {placeholder}")
        if "{samples}" not in self.verify_with_samples_template:
            raise ValueError("verify_with_samples_template must contain {samples}")
        return self


class Config(Frozen):
    """Top-level config. One YAML file, fully validated."""

    run_name: str
    model_under_test: ModelSpec
    greedy: GreedySpec
    sampling: SamplingSpec
    nli: NLISpec
    signals: SignalsSpec = SignalsSpec()
    dataset_mix: DatasetMix
    difficulty: DifficultySpec = DifficultySpec()
    few_shot: FewShotSpec = FewShotSpec()
    dataset_seed: int = 12345
    pilot_gate: PilotGateSpec
    judges: JudgeSpec
    split: SplitSpec
    analysis: AnalysisSpec
    paths: PathsSpec = PathsSpec()
    prompts: PromptSpec
    is_pilot: bool = False

    @model_validator(mode="after")
    def _judges_differ_from_subject(self) -> Config:
        # A model judging its own output inherits its own blind spots.
        subject = self.model_under_test.name
        if self.judges.primary.name == subject:
            raise ValueError(f"primary judge must differ from model under test ({subject})")
        if self.judges.secondary.name == subject:
            raise ValueError(f"secondary judge must differ from model under test ({subject})")
        return self

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} did not parse to a mapping")
        return cls.model_validate(raw)
