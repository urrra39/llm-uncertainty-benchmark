"""`configs/run3_gpu.yaml` is the GPU run's contract. Pin what it must say.

Run #3 has not been executed. These tests do not check any result; they check
that the config a future GPU run will load says the four things the run depends
on, so a later edit cannot quietly change one of them:

- generation batch size is 1: defect D27 is open (the CPU/bfloat16
  bit-identity measurement does not transfer to CUDA/fp16, and per-forward
  reseeding moves sampled tokens), so family A is only valid unbatched until
  the batch-invariance harness passes on the target device;
- the model is loaded unquantized, for the same reason one level up;
- the dataset split is 300 PopQA / 300 TriviaQA, which is what run #2's 90/30
  imbalance cost it;
- per-dataset bootstrap intervals are switched on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from unc_bench.analysis.extended import per_dataset_auroc
from unc_bench.config import Config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
RUN3 = "run3_gpu.yaml"

#: Every key `ModelSpec` accepts that could request a reduced-precision or
#: quantized load. `dtype` is the only one, and fp16/bf16/fp32 are all exact
#: forward passes rather than post-training quantization. Named here so that if
#: a quantization field is ever added to `ModelSpec`, this list is the place the
#: test starts failing.
UNQUANTIZED_DTYPES = ("float32", "bfloat16", "float16")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load(CONFIG_DIR / RUN3)


def test_run3_config_loads(cfg: Config) -> None:
    # The whole point of the Pydantic layer is that a config which would produce
    # a wrong analysis fails at load. This asserts the file gets through it.
    assert cfg.run_name == "run3_gpu_balanced"
    assert cfg.greedy.temperature == 0.0
    assert cfg.dataset_mix.total == 600


def test_run3_generation_batch_size_is_one_while_d27_is_open(cfg: Config) -> None:
    # D27: padding a ragged batch perturbs per-token logprobs by up to 2.52e-02
    # against the unbatched values on CPU/bfloat16, and that measurement does
    # not transfer to CUDA/fp16. Length-bucketing plus per-prompt seeds narrow
    # the exposure but do not close it, so the run stays at width 1 until the
    # batch-invariance harness passes on the T4 in fp16.
    assert cfg.model_under_test.generation_batch_size == 1


def test_run3_config_text_points_at_d27_and_its_status(cfg: Config) -> None:
    # The number above is only safe while the reason for it is written next to
    # it — the defect, what was measured where, and what would unpin it.
    text = (CONFIG_DIR / RUN3).read_text(encoding="utf-8")
    assert "D27" in text
    assert "nondeterminism" in text
    assert "generation_batch_size" in text


def test_run3_model_is_unquantized(cfg: Config) -> None:
    # Signal family A reads exact per-token logprobs off the generation scores,
    # and post-training quantization perturbs the logits they come from. Family A
    # is nine of the eighteen distinct orderings, so a quantized load would
    # compromise half the study.
    spec = cfg.model_under_test
    assert spec.name == "Qwen/Qwen2.5-3B-Instruct"
    assert spec.dtype in UNQUANTIZED_DTYPES
    assert spec.dtype == "float16"
    # There is no quantization knob to set. `ModelSpec` forbids extra keys, so a
    # config asking for one fails to load; this asserts the field set stays that
    # way rather than gaining a quantization option that defaults to on.
    assert "quantization" not in type(spec).model_fields
    assert not any("quant" in name for name in type(spec).model_fields)


def test_run3_split_is_300_300(cfg: Config) -> None:
    mix = cfg.dataset_mix
    assert (mix.popqa, mix.triviaqa, mix.simpleqa) == (300, 300, 0)
    assert mix.popqa == mix.triviaqa


def test_run3_popqa_filter_is_seven_relations_without_the_inverse(
    cfg: Config,
) -> None:
    # Part A1 + C2: `capital of` removed (echo pathology),
    # `place of birth` removed (granularity-span 0.620 in
    # data/gold_quality_report.json), `genre` added (single-alias 0.074).
    # `religion` and `occupation` were flagged by suspicion and cleared by
    # measurement. Quantile 0.5 buys the 3x sampling margin (see below).
    relations = cfg.difficulty.popqa_relations
    assert relations is not None
    assert set(relations) == {
        "capital",
        "country",
        "sport",
        "color",
        "religion",
        "occupation",
        "genre",
    }
    assert "capital of" not in relations
    assert "place of birth" not in relations
    assert cfg.difficulty.popqa_popularity_quantile == 0.5
    assert cfg.difficulty.allow_inverse_relations is False
    assert cfg.difficulty.drop_gold_in_question is True
    # TriviaQA is unchanged from run #2: easy_only + min_aliases 20.
    assert cfg.difficulty.triviaqa_min_aliases == 20
    assert cfg.difficulty.triviaqa_easy_only is True


def test_run3_per_dataset_bootstrap_is_on(cfg: Config) -> None:
    """Per-dataset intervals are enabled by the analysis block, not a flag.

    `per_dataset_auroc` computes a within-subset bootstrap whenever it is handed
    a config, and `report.py` always hands it one. So "bootstrap is on" is two
    claims: the resample count is the full one, and driving the function with
    this config really does produce an interval. Both are checked, because the
    first alone would pass on a config whose analysis block was fine but whose
    intervals never got computed.
    """
    assert cfg.analysis.bootstrap_resamples == 10_000
    assert cfg.analysis.ci_level == 0.95

    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "dataset": ["popqa"] * n + ["triviaqa"] * n,
            "s": rng.normal(size=2 * n),
        }
    )
    y = np.array([True, False] * n, dtype=bool)

    out = per_dataset_auroc(frame, ["s"], y, cfg)
    assert out["available"] is True
    assert out["ci_available"] is True
    assert out["ci_resamples"] == 10_000
    assert out["ci_level"] == 0.95
    assert out["ci_seed"] == cfg.analysis.bootstrap_seed
    for source in ("popqa", "triviaqa"):
        entry = out["datasets"][source]["signals"]["s"]
        assert "auroc_ci" in entry
        assert "auprc_ci" in entry
        assert entry["excludes_chance"] in (True, False)

    # And without a config there is no interval, which is what makes the
    # assertion above mean something.
    without = per_dataset_auroc(frame, ["s"], y, None)
    assert without["ci_available"] is False
    assert "auroc_ci" not in without["datasets"]["popqa"]["signals"]["s"]


def test_run3_writes_its_own_artifacts_and_results(cfg: Config) -> None:
    # Run #2's results.json is frozen. Run #3 must not be able to overwrite it.
    assert cfg.paths.results_json == Path("results_run3.json")
    assert cfg.paths.artifacts_dir == Path("data/run3")
    run2 = Config.load(CONFIG_DIR / "run2.yaml")
    assert cfg.paths.results_json != run2.paths.results_json
    assert cfg.paths.artifacts_dir != run2.paths.artifacts_dir
    assert cfg.paths.figures_dir != run2.paths.figures_dir


def test_run3_inherits_run2s_frozen_blocks(cfg: Config) -> None:
    # Run #3 is meant to differ from run #2 in model size, n and dataset balance
    # and in nothing else. The prompt block in particular is part of the cache
    # key, and a drift there would make the two runs incomparable.
    run2 = Config.load(CONFIG_DIR / "run2.yaml")
    assert cfg.prompts == run2.prompts
    assert cfg.greedy == run2.greedy
    assert cfg.sampling == run2.sampling
    assert cfg.nli == run2.nli
    assert cfg.few_shot == run2.few_shot
    assert cfg.judges == run2.judges
    assert cfg.split == run2.split
    assert cfg.dataset_seed == run2.dataset_seed
    # The analysis block matches except the cluster-bootstrap switch: with
    # dedup active the clusters are singletons and the two draws agree exactly
    # at a fixed seed, so this is defence in depth rather than a new estimator.
    assert cfg.analysis.model_copy(update={"cluster_bootstrap": False}) == run2.analysis
    assert cfg.analysis.cluster_bootstrap is True
    assert run2.analysis.cluster_bootstrap is False


def test_run3_judges_differ_from_the_new_subject(cfg: Config) -> None:
    # The subject model changed; the judge guard has to still hold against the
    # new name.
    assert cfg.judges.primary.name != cfg.model_under_test.name
    assert cfg.judges.secondary.name != cfg.model_under_test.name
