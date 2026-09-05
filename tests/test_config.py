"""The config validators exist to stop specific mistakes. Test that they do."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from unc_bench.config import Config, GreedySpec, PromptSpec, SamplingSpec, SplitSpec

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _raw(name: str) -> dict[str, Any]:
    loaded = yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("name", ["default.yaml", "pilot.yaml", "run2.yaml", "run3_gpu.yaml"])
def test_shipped_configs_validate(name: str) -> None:
    cfg = Config.load(CONFIG_DIR / name)
    assert cfg.greedy.temperature == 0.0
    assert cfg.dataset_mix.total > 0


def test_default_mix_is_triviaqa_only() -> None:
    # The shipped run is TriviaQA-only: the SimpleQA builder does not exist and
    # PopQA is held back. A nonzero count for either would ask the pipeline for
    # questions it cannot draw.
    mix = Config.load(CONFIG_DIR / "default.yaml").dataset_mix
    assert (mix.popqa, mix.triviaqa, mix.simpleqa) == (0, 100, 0)
    assert mix.total == 100


def test_pilot_mix_is_40_triviaqa() -> None:
    mix = Config.load(CONFIG_DIR / "pilot.yaml").dataset_mix
    assert (mix.popqa, mix.triviaqa, mix.simpleqa) == (0, 40, 0)
    assert mix.total == 40


def test_pilot_gate_n_matches_the_pilot_mix() -> None:
    # The gate reports rates over whatever the pilot generated, so a gate
    # configured for a different n than the mix draws would describe a sample
    # size that was never run.
    cfg = Config.load(CONFIG_DIR / "pilot.yaml")
    assert cfg.pilot_gate.n_questions == cfg.dataset_mix.total


def test_ece_bins_are_ten_in_every_shipped_config() -> None:
    # The README's calibration section states 10 bins. It reads one number and
    # applies it to the whole repository, so every config that ships has to say
    # 10 or that sentence is wrong for one of them. Enumerated from the
    # directory rather than listed, so a new config is covered on arrival.
    names = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    assert len(names) >= 2
    for name in names:
        assert Config.load(CONFIG_DIR / name).analysis.ece_bins == 10, name


def test_unknown_key_is_rejected() -> None:
    # A typo in YAML must fail loudly rather than be ignored.
    raw = _raw("default.yaml")
    raw["bootstrap_resamples"] = 10  # right name, wrong nesting level
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate(raw)


def test_greedy_spec_refuses_nonzero_temperature() -> None:
    # Trap: computing signal family A on a sampled generation.
    with pytest.raises(ValidationError, match="temperature == 0.0"):
        GreedySpec(temperature=0.7)


def test_sampling_spec_refuses_zero_temperature() -> None:
    with pytest.raises(ValidationError, match="must be > 0"):
        SamplingSpec(temperature=0.0)


def test_ablation_cannot_exceed_n_samples() -> None:
    with pytest.raises(ValidationError, match="exceeds n_samples"):
        SamplingSpec(n_samples=3, ablation_n=(1, 2, 3, 5))


def test_sampling_seeds_are_distinct() -> None:
    spec = SamplingSpec()
    seeds = [spec.seed_for(i) for i in range(spec.n_samples)]
    assert len(set(seeds)) == spec.n_samples


def test_question_seeds_depend_on_qid_and_index() -> None:
    spec = SamplingSpec()
    assert spec.seed_for_question(0, "popqa-1") == spec.seed_for_question(0, "popqa-1")
    assert spec.seed_for_question(0, "popqa-1") != spec.seed_for_question(0, "popqa-2")
    assert spec.seed_for_question(0, "popqa-1") != spec.seed_for_question(1, "popqa-1")


def test_judge_may_not_be_the_model_under_test() -> None:
    raw = _raw("default.yaml")
    raw["judges"]["primary"]["name"] = raw["model_under_test"]["name"]
    with pytest.raises(ValidationError, match="primary judge must differ"):
        Config.model_validate(raw)


def test_secondary_judge_may_not_be_the_model_under_test() -> None:
    raw = _raw("default.yaml")
    raw["judges"]["secondary"]["name"] = raw["model_under_test"]["name"]
    with pytest.raises(ValidationError, match="secondary judge must differ"):
        Config.model_validate(raw)


def test_split_fraction_bounds() -> None:
    with pytest.raises(ValidationError, match="not a useful split"):
        SplitSpec(train_fraction=0.9)


def test_prompt_template_placeholders_are_enforced() -> None:
    good = _raw("default.yaml")["prompts"]

    bad = copy.deepcopy(good)
    bad["user_template"] = "A:"
    with pytest.raises(ValidationError, match=r"\{question\}"):
        PromptSpec.model_validate(bad)

    bad = copy.deepcopy(good)
    bad["verify_template"] = "Is {question} right?"
    with pytest.raises(ValidationError, match=r"\{answer\}"):
        PromptSpec.model_validate(bad)

    bad = copy.deepcopy(good)
    bad["verify_with_samples_template"] = "Q {question} A {answer}"
    with pytest.raises(ValidationError, match=r"\{samples\}"):
        PromptSpec.model_validate(bad)


def test_config_is_frozen() -> None:
    cfg = Config.load(CONFIG_DIR / "default.yaml")
    # The attribute name is built at runtime so that mypy (which rejects a
    # direct assignment to a frozen field, correctly) and ruff (which rejects a
    # setattr with a literal name) both stay quiet. The runtime guard is what is
    # under test here.
    field = "dataset" + "_seed"
    with pytest.raises(ValidationError):
        setattr(cfg, field, 7)


def test_prompt_instructs_shortest_span_and_unknown() -> None:
    prompts = Config.load(CONFIG_DIR / "default.yaml").prompts
    assert "shortest" in prompts.system.lower()
    assert prompts.abstain_token in prompts.system
