"""Family B, driven by a scripted entailment model.

Every clustering test here uses `ScriptedEntailmentModel` rather than real
weights. The reason is not speed, though CI installing neither torch nor
transformers is a real constraint. The reason is that the property under test is
an asymmetry, and you cannot construct a guaranteed-asymmetric pair against
weights you have not pinned. "Paris" entails "Paris, France" at about 0.99 with
essentially nothing coming back the other way — true of the base checkpoint
today, not something a test can assert. Scripting the scores makes
"bidirectional means bidirectional" a property of the code rather than a property
of a download.

The entropy targets are arithmetic: 0 for one cluster, ln2 for two equal, ln6 for
six singletons, 0.6365142 for a 4/2 split, 1.0114043 for 3/2/1.
"""

from __future__ import annotations

import math

import pytest

from unc_bench.config import NLISpec
from unc_bench.signals.base import FAMILY_B, orient_all, signal_names
from unc_bench.signals.consistency import (
    DISAGREEMENT_RATE,
    DISTINCT_COUNT,
    DISTINCT_FRACTION,
    FAMILY_B_SIGNALS,
    MEAN_PAIRWISE_F1,
    SEMANTIC_ENTROPY,
    SEMANTIC_ENTROPY_NORM,
    cluster_answers,
    cluster_entropy,
    compute_family_b,
)
from unc_bench.signals.nli import ScriptedEntailmentModel, resolve_entailment_index

SPEC = NLISpec()


def _sizes(clusters: list[list[int]]) -> list[int]:
    return sorted((len(c) for c in clusters), reverse=True)


# ----------------------------------------------------- entailment index


def test_entailment_index_resolved_by_name_not_position() -> None:
    forward = {0: "entailment", 1: "neutral", 2: "contradiction"}
    reverse = {0: "contradiction", 1: "neutral", 2: "entailment"}
    assert resolve_entailment_index(forward) == 0
    assert resolve_entailment_index(reverse) == 2


def test_entailment_index_is_case_and_whitespace_insensitive() -> None:
    assert resolve_entailment_index({0: "NEUTRAL", 1: " Entailment "}) == 1


def test_entailment_index_raises_rather_than_guessing() -> None:
    """A wrong index is undetectable downstream, so refuse to guess.

    Every number stays in range and every cluster stays plausible; only the
    conclusions invert. Guessing 0 here would be the single most expensive
    silent bug available in this project.
    """
    with pytest.raises(ValueError, match="refusing to guess"):
        resolve_entailment_index({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


# --------------------------------------------------------- clustering


def test_clustering_requires_entailment_in_both_directions() -> None:
    """The Paris case. One-directional entailment must NOT merge.

    "paris" entails "paris france" — anyone who says Paris has said something a
    Frenchman would confirm — but "paris france" does not entail "paris" as a
    claim about which Paris. Real weights score the forward direction ~0.99 and
    the reverse near zero. Scripted here so the assertion is about the code.
    """
    model = ScriptedEntailmentModel(
        {
            ("paris", "paris france"): 0.99,
            ("paris france", "paris"): 0.02,
        }
    )
    clusters = cluster_answers(["Paris", "Paris, France"], model, SPEC)
    assert _sizes(clusters) == [1, 1]


def test_clustering_merges_when_both_directions_pass() -> None:
    model = ScriptedEntailmentModel(
        {
            ("nyc", "new york city"): 0.96,
            ("new york city", "nyc"): 0.94,
        }
    )
    clusters = cluster_answers(["NYC", "New York City"], model, SPEC)
    assert _sizes(clusters) == [2]


def test_clustering_respects_the_configured_threshold() -> None:
    """Note the fixture strings: real words, no single letters.

    My first version used "a b" and "c d" as filler and the test failed with
    both answers in singletons. The cause was the fixture, not the threshold:
    "a" is an article and normalizes away, so the scripted key ("a b", "c d")
    never matched the ("b", "d") pair the clustering actually asked about, and
    the model returned its 0.0 default. The same trap already cost me two
    normalization tests in an earlier session.
    """
    scores = {("rome", "cairo"): 0.6, ("cairo", "rome"): 0.6}
    lenient = NLISpec(entailment_threshold=0.5)
    strict = NLISpec(entailment_threshold=0.7)
    pair = ["Rome", "Cairo"]
    assert _sizes(cluster_answers(pair, ScriptedEntailmentModel(scores), lenient)) == [2]
    assert _sizes(cluster_answers(pair, ScriptedEntailmentModel(scores), strict)) == [1, 1]


def test_exact_duplicates_never_reach_the_nli_model() -> None:
    """Identical answers must cluster without asking, because MNLI is unreliable
    at entailing a string against itself.

    The scripted model defaults to 0.0, so if these five identical answers went
    through it they would split into five singletons and semantic entropy would
    read ln5 — the maximum — for the model's most confident possible output.
    """
    model = ScriptedEntailmentModel({}, default=0.0)
    clusters = cluster_answers(["Rome"] * 5, model, SPEC)
    assert _sizes(clusters) == [5]
    assert model.calls == []


def test_normalization_is_applied_before_duplicate_matching() -> None:
    model = ScriptedEntailmentModel({}, default=0.0)
    clusters = cluster_answers(["The Beatles", "the beatles.", "  THE BEATLES  "], model, SPEC)
    assert _sizes(clusters) == [3]
    assert model.calls == []


def test_empty_answers_cluster_together_rather_than_splitting() -> None:
    """An empty string entails nothing, so it needs the duplicate path.

    Without it, a model that emitted nothing five times would get maximum
    semantic entropy for producing the same output every time.
    """
    model = ScriptedEntailmentModel({}, default=0.0)
    assert _sizes(cluster_answers(["", "", ""], model, SPEC)) == [3]


def test_unrelated_answers_stay_in_singletons() -> None:
    model = ScriptedEntailmentModel({}, default=0.05)
    assert _sizes(cluster_answers(["Rome", "Cairo", "Lima"], model, SPEC)) == [1, 1, 1]


# ------------------------------------------------------------- entropy


def test_entropy_of_one_cluster_is_zero() -> None:
    assert cluster_entropy([[0, 1, 2, 3, 4, 5]]) == pytest.approx(0.0)


def test_entropy_of_two_equal_clusters_is_ln2() -> None:
    assert cluster_entropy([[0, 1, 2], [3, 4, 5]]) == pytest.approx(math.log(2))


def test_entropy_of_six_singletons_is_ln6() -> None:
    assert cluster_entropy([[i] for i in range(6)]) == pytest.approx(math.log(6))


def test_entropy_of_a_four_two_split_matches_hand_computed() -> None:
    # -(4/6 ln(4/6) + 2/6 ln(2/6)) = 0.6365141682948128
    got = cluster_entropy([[0, 1, 2, 3], [4, 5]])
    assert got == pytest.approx(0.6365141682948128, abs=1e-9)


def test_entropy_of_a_three_two_one_split_matches_hand_computed() -> None:
    # -(3/6 ln(3/6) + 2/6 ln(2/6) + 1/6 ln(1/6)) = 1.0114042647073518
    got = cluster_entropy([[0, 1, 2], [3, 4], [5]])
    assert got == pytest.approx(1.0114042647073518, abs=1e-9)


def test_entropy_of_no_clusters_is_nan() -> None:
    assert math.isnan(cluster_entropy([]))


# -------------------------------------------------------- family B rows


def _agreeing_model() -> ScriptedEntailmentModel:
    return ScriptedEntailmentModel({}, default=0.0)


def test_greedy_answer_is_element_zero_of_the_answer_set() -> None:
    """Six answers, not five: the greedy answer participates in the clustering.

    Five distinct samples plus a distinct greedy answer is six singletons, so
    the entropy target is ln6. If the greedy answer were excluded it would be
    ln5 and the signal would be describing a population that does not include
    the answer being scored.
    """
    out = compute_family_b(
        "Rome",
        ["Cairo", "Lima", "Oslo", "Bern", "Kiev"],
        _agreeing_model(),
        SPEC,
    )
    assert out[SEMANTIC_ENTROPY.name] == pytest.approx(math.log(6))
    assert out[DISTINCT_COUNT.name] == pytest.approx(6.0)


def test_unanimous_answer_set_is_maximally_consistent() -> None:
    out = compute_family_b("Rome", ["Rome"] * 5, _agreeing_model(), SPEC)
    assert out[DISAGREEMENT_RATE.name] == pytest.approx(0.0)
    assert out[DISTINCT_COUNT.name] == pytest.approx(1.0)
    assert out[DISTINCT_FRACTION.name] == pytest.approx(1 / 6)
    assert out[MEAN_PAIRWISE_F1.name] == pytest.approx(1.0)
    assert out[SEMANTIC_ENTROPY.name] == pytest.approx(0.0)
    assert out[SEMANTIC_ENTROPY_NORM.name] == pytest.approx(0.0)


def test_disagreement_rate_excludes_the_greedy_self_comparison() -> None:
    """Two of five samples differ, so the rate is 2/5 and not 2/6.

    Including the greedy answer's trivial agreement with itself would compress
    the range by (N+1)/N and cap the signal below 1.0 for no information.
    """
    out = compute_family_b(
        "Rome",
        ["Rome", "Rome", "Rome", "Cairo", "Lima"],
        _agreeing_model(),
        SPEC,
    )
    assert out[DISAGREEMENT_RATE.name] == pytest.approx(2 / 5)


def test_normalized_entropy_is_on_the_unit_interval() -> None:
    out = compute_family_b(
        "Rome",
        ["Cairo", "Lima", "Oslo", "Bern", "Kiev"],
        _agreeing_model(),
        SPEC,
    )
    assert out[SEMANTIC_ENTROPY_NORM.name] == pytest.approx(1.0)


def test_no_samples_gives_nan_not_zero() -> None:
    out = compute_family_b("Rome", [], _agreeing_model(), SPEC)
    assert set(out) == {s.name for s in FAMILY_B_SIGNALS}
    for name, value in out.items():
        assert math.isnan(value), f"{name} returned {value!r} with no samples"


# --------------------------------------------------------- orientation


def test_every_family_b_signal_is_registered_in_family_b() -> None:
    from unc_bench.signals.consistency import FAMILY_B_SAMPLES_ONLY_SIGNALS

    assert set(signal_names(FAMILY_B)) == {
        s.name for s in (*FAMILY_B_SIGNALS, *FAMILY_B_SAMPLES_ONLY_SIGNALS)
    }


def test_oriented_family_b_values_all_increase_with_wrongness() -> None:
    consistent = orient_all(compute_family_b("Rome", ["Rome"] * 5, _agreeing_model(), SPEC))
    scattered = orient_all(
        compute_family_b("Rome", ["Cairo", "Lima", "Oslo", "Bern", "Kiev"], _agreeing_model(), SPEC)
    )
    for name in consistent:
        assert (
            consistent[name] < scattered[name]
        ), f"{name} is oriented backwards: {consistent[name]} !< {scattered[name]}"


def test_samples_only_excludes_the_greedy_answer() -> None:
    """Five distinct samples plus a distinct greedy answer: samples-only sees
    five singletons (ln5), while the greedy-included set sees six (ln6)."""
    from unc_bench.signals.consistency import (
        DISTINCT_COUNT_SO,
        SEMANTIC_ENTROPY_SO,
        compute_family_b_samples_only,
    )

    out = compute_family_b_samples_only(
        ["Cairo", "Lima", "Oslo", "Bern", "Kiev"], _agreeing_model(), SPEC
    )
    assert out[SEMANTIC_ENTROPY_SO.name] == pytest.approx(math.log(5))
    assert out[DISTINCT_COUNT_SO.name] == pytest.approx(5.0)


def test_samples_only_disagreement_is_against_the_plurality() -> None:
    from unc_bench.signals.consistency import (
        DISAGREEMENT_RATE_SO,
        FAMILY_B_SAMPLES_ONLY_SIGNALS,
        MEAN_PAIRWISE_F1_SO,
        compute_family_b_samples_only,
    )

    out = compute_family_b_samples_only(
        ["Rome", "Rome", "Rome", "Cairo", "Lima"], _agreeing_model(), SPEC
    )
    assert out[DISAGREEMENT_RATE_SO.name] == pytest.approx(2 / 5)
    assert set(out) == {s.name for s in FAMILY_B_SAMPLES_ONLY_SIGNALS}
    assert out[MEAN_PAIRWISE_F1_SO.name] > 0.0


def test_samples_only_unanimous_is_maximally_consistent() -> None:
    from unc_bench.signals.consistency import (
        DISAGREEMENT_RATE_SO,
        DISTINCT_COUNT_SO,
        SEMANTIC_ENTROPY_SO,
        compute_family_b_samples_only,
    )

    out = compute_family_b_samples_only(["Rome"] * 5, _agreeing_model(), SPEC)
    assert out[DISAGREEMENT_RATE_SO.name] == pytest.approx(0.0)
    assert out[DISTINCT_COUNT_SO.name] == pytest.approx(1.0)
    assert out[SEMANTIC_ENTROPY_SO.name] == pytest.approx(0.0)


def test_samples_only_empty_is_nan_not_zero() -> None:
    from unc_bench.signals.consistency import (
        FAMILY_B_SAMPLES_ONLY_SIGNALS,
        compute_family_b_samples_only,
    )

    out = compute_family_b_samples_only([], _agreeing_model(), SPEC)
    assert set(out) == {s.name for s in FAMILY_B_SAMPLES_ONLY_SIGNALS}
    for name, value in out.items():
        assert math.isnan(value), f"{name} returned {value!r} with no samples"


def test_exhaustive_clustering_closes_transitive_chains() -> None:
    """A entails B and B entails C, but A does not directly entail C.

    Greedy assignment checks C against A's cluster representative only and
    leaves C alone; the transitive closure merges all three. This is the case
    the greedy order-dependence audit exists to catch."""
    from unc_bench.signals.consistency import cluster_answers_exhaustive

    scores = {
        ("ax", "bx"): 0.9,
        ("bx", "ax"): 0.9,
        ("bx", "cx"): 0.9,
        ("cx", "bx"): 0.9,
        ("ax", "cx"): 0.1,
        ("cx", "ax"): 0.1,
    }
    model = ScriptedEntailmentModel(scores, default=0.0)
    assert _sizes(cluster_answers(["ax", "bx", "cx"], model, SPEC)) == [2, 1]
    assert _sizes(cluster_answers_exhaustive(["ax", "bx", "cx"], model, SPEC)) == [3]


def test_exhaustive_agrees_with_greedy_on_unambiguous_sets() -> None:
    from unc_bench.signals.consistency import cluster_answers_exhaustive

    model = ScriptedEntailmentModel({}, default=0.0)
    answers = ["Rome", "Rome", "Cairo", "Lima"]
    assert _sizes(cluster_answers(answers, model, SPEC)) == [2, 1, 1]
    assert _sizes(cluster_answers_exhaustive(answers, model, SPEC)) == [2, 1, 1]


def test_exhaustive_merges_duplicates_without_model_calls() -> None:
    from unc_bench.signals.consistency import cluster_answers_exhaustive

    model = ScriptedEntailmentModel({}, default=0.0)
    assert _sizes(cluster_answers_exhaustive(["x", "x", "x"], model, SPEC)) == [3]
    assert model.calls == []


def test_clustering_audit_flags_order_dependent_rows() -> None:
    from unc_bench.signals.consistency import clustering_disagreement

    scores = {
        ("ax", "bx"): 0.9,
        ("bx", "ax"): 0.9,
        ("bx", "cx"): 0.9,
        ("cx", "bx"): 0.9,
        ("ax", "cx"): 0.1,
        ("cx", "ax"): 0.1,
    }
    assert (
        clustering_disagreement("ax", ["bx", "cx"], ScriptedEntailmentModel(scores), SPEC) is True
    )
    agreeing = ScriptedEntailmentModel({}, default=0.0)
    assert clustering_disagreement("Rome", ["Rome", "Rome"], agreeing, SPEC) is False
