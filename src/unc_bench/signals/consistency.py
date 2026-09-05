"""Signal family B: self-consistency over N sampled answers.

Cost is about 6x: one greedy answer plus N samples. The greedy answer is element
0 of the answer set by construction, so the samples are extra evidence about the
answer being scored rather than a separate population.

Semantic entropy follows Farquhar et al., Nature 2024 (Kuhn et al. 2023 for the
clustering): group answers into meaning classes by bidirectional entailment, then
take the Shannon entropy of the cluster-size distribution. The bidirectional
requirement is the whole content of the method. One-directional entailment merges
"Paris" into "Paris, France" and then reports the pair as one meaning, which
throws away exactly the distinction the signal is supposed to detect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from unc_bench.config import NLISpec
from unc_bench.normalize import (
    count_distinct,
    mean_pairwise_token_f1,
    normalize_answer,
)
from unc_bench.signals.base import (
    FAMILY_B,
    ORIENT_CONFIDENCE,
    ORIENT_RISK,
    SignalSpec,
    register,
)
from unc_bench.signals.nli import EntailmentModel

DISAGREEMENT_RATE = register(
    SignalSpec(
        name="b_disagreement_rate",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="fraction of samples whose normalized answer differs from greedy",
    )
)
DISTINCT_COUNT = register(
    SignalSpec(
        name="b_distinct_count",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="number of distinct normalized answers in the answer set",
    )
)
DISTINCT_FRACTION = register(
    SignalSpec(
        name="b_distinct_fraction",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="distinct count divided by answer-set size",
        rank_equivalent_to="b_distinct_count",
    )
)
MEAN_PAIRWISE_F1 = register(
    SignalSpec(
        name="b_mean_pairwise_f1",
        family=FAMILY_B,
        orientation=ORIENT_CONFIDENCE,
        description="mean token F1 over all unordered pairs in the answer set",
    )
)
SEMANTIC_ENTROPY = register(
    SignalSpec(
        name="b_semantic_entropy",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="Shannon entropy over bidirectional-entailment clusters",
    )
)
SEMANTIC_ENTROPY_NORM = register(
    SignalSpec(
        name="b_semantic_entropy_normalized",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="semantic entropy divided by ln(answer-set size)",
        rank_equivalent_to="b_semantic_entropy",
    )
)

#: Suffix for the samples-only variants below.
SAMPLES_ONLY_SUFFIX = "_samples_only"

DISAGREEMENT_RATE_SO = register(
    SignalSpec(
        name=f"b_disagreement_rate{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="fraction of samples differing from the plurality sample answer",
    )
)
DISTINCT_COUNT_SO = register(
    SignalSpec(
        name=f"b_distinct_count{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="distinct normalized answers among the samples alone",
    )
)
DISTINCT_FRACTION_SO = register(
    SignalSpec(
        name=f"b_distinct_fraction{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="samples-only distinct count divided by sample count",
        rank_equivalent_to=f"b_distinct_count{SAMPLES_ONLY_SUFFIX}",
    )
)
MEAN_PAIRWISE_F1_SO = register(
    SignalSpec(
        name=f"b_mean_pairwise_f1{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_CONFIDENCE,
        description="mean token F1 over unordered sample pairs, greedy excluded",
    )
)
SEMANTIC_ENTROPY_SO = register(
    SignalSpec(
        name=f"b_semantic_entropy{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="Shannon entropy over bidirectional-entailment clusters of the samples alone",
    )
)
SEMANTIC_ENTROPY_NORM_SO = register(
    SignalSpec(
        name=f"b_semantic_entropy_normalized{SAMPLES_ONLY_SUFFIX}",
        family=FAMILY_B,
        orientation=ORIENT_RISK,
        description="samples-only semantic entropy divided by ln(sample count)",
        rank_equivalent_to=f"b_semantic_entropy{SAMPLES_ONLY_SUFFIX}",
    )
)

FAMILY_B_SIGNALS: tuple[SignalSpec, ...] = (
    DISAGREEMENT_RATE,
    DISTINCT_COUNT,
    DISTINCT_FRACTION,
    MEAN_PAIRWISE_F1,
    SEMANTIC_ENTROPY,
    SEMANTIC_ENTROPY_NORM,
)

#: The samples-only variant family: the same six statistics over the stochastic
#: samples alone, without the greedy answer. This is the variant comparable to
#: published semantic-entropy work, which estimates one output distribution at
#: one temperature; the greedy-included set above additionally scores the answer
#: under evaluation and is reported beside it so the mixing bias is measured
#: rather than described (LIMITATIONS item 15).
FAMILY_B_SAMPLES_ONLY_SIGNALS: tuple[SignalSpec, ...] = (
    DISAGREEMENT_RATE_SO,
    DISTINCT_COUNT_SO,
    DISTINCT_FRACTION_SO,
    MEAN_PAIRWISE_F1_SO,
    SEMANTIC_ENTROPY_SO,
    SEMANTIC_ENTROPY_NORM_SO,
)


def cluster_answers(
    answers: Sequence[str],
    model: EntailmentModel,
    spec: NLISpec,
) -> list[list[int]]:
    """Group answer indices into meaning clusters by bidirectional entailment.

    Greedy single-pass assignment against each cluster's representative, which is
    the standard formulation. It is order-dependent in principle; in practice the
    answer sets here are 5 or 6 short spans and the representative is the first
    member, so the greedy pass and an exhaustive one agree on every case observed.

    Two shortcuts, both deliberate:

    Exact normalized duplicates join without consulting the model. An MNLI
    checkpoint does not reliably entail a string against itself, and letting that
    noise into the clustering would sometimes split five identical samples into
    five clusters — handing the highest semantic entropy to the most confident
    items, which inverts the signal on exactly the rows it should be surest about.

    Empty answers cluster together as one meaning. An empty string entails
    nothing, so without this every empty answer becomes its own singleton and a
    model that emitted nothing five times would score maximum entropy.
    """
    clusters: list[list[int]] = []
    representatives: list[str] = []
    normalized = [normalize_answer(a) for a in answers]

    for index, text in enumerate(normalized):
        placed = False
        # Cheap path first: exact normalized match against any representative.
        for cluster_index, rep in enumerate(representatives):
            if rep == text:
                clusters[cluster_index].append(index)
                placed = True
                break
        if placed:
            continue

        # Ask the model both directions against every representative at once.
        pairs: list[tuple[str, str]] = []
        for rep in representatives:
            pairs.append((rep, text))
            pairs.append((text, rep))
        probs = model.entailment_probs(pairs) if pairs else []
        for cluster_index in range(len(representatives)):
            forward = probs[2 * cluster_index]
            backward = probs[2 * cluster_index + 1]
            # BOTH directions, or it is not the same meaning.
            if forward >= spec.entailment_threshold and backward >= spec.entailment_threshold:
                clusters[cluster_index].append(index)
                placed = True
                break
        if not placed:
            clusters.append([index])
            representatives.append(text)

    return clusters


def cluster_entropy(clusters: Sequence[Sequence[int]]) -> float:
    """Shannon entropy in nats of the cluster-size distribution.

    The discrete estimator, matching the Farquhar et al. formulation for
    sampled answers with no per-sequence likelihoods available.
    """
    total = sum(len(c) for c in clusters)
    if total <= 0:
        return float("nan")
    out = 0.0
    for cluster in clusters:
        p = len(cluster) / total
        if p > 0.0:
            out -= p * math.log(p)
    return out


def cluster_answers_exhaustive(
    answers: Sequence[str],
    model: EntailmentModel,
    spec: NLISpec,
) -> list[list[int]]:
    """Transitive closure over the bidirectional-entailment graph.

    Every unordered pair is scored in both directions; an edge joins two
    answers only when both directions clear the threshold, and clusters are
    the connected components. This is the audit for `cluster_answers`' greedy
    single-pass assignment against first-member representatives, which is
    order-dependent in principle: where the two disagree, the exhaustive
    partition is primary. Exact normalized duplicates (and empties) merge
    without consulting the model, for the same reason as in the greedy pass —
    an MNLI checkpoint does not reliably entail a string against itself.
    """
    normalized = [normalize_answer(a) for a in answers]
    parent = list(range(len(normalized)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for index, text in enumerate(normalized):
        for other in range(index):
            if normalized[other] == text:
                union(index, other)
    pairs: list[tuple[str, str]] = []
    pair_index: list[tuple[int, int]] = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            if find(i) == find(j):
                continue
            pairs.append((normalized[i], normalized[j]))
            pair_index.append((i, j))
            pairs.append((normalized[j], normalized[i]))
            pair_index.append((j, i))
    if pairs:
        probs = model.entailment_probs(pairs)
        for step in range(0, len(pairs), 2):
            i, j = pair_index[step]
            if find(i) == find(j):
                continue
            if (
                probs[step] >= spec.entailment_threshold
                and probs[step + 1] >= spec.entailment_threshold
            ):
                union(i, j)

    grouped: dict[int, list[int]] = {}
    for index in range(len(normalized)):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for members in grouped.values()]


def clustering_disagreement(
    greedy_answer: str,
    sample_answers: Sequence[str],
    model: EntailmentModel,
    spec: NLISpec,
) -> bool:
    """Whether greedy and exhaustive clustering partition one answer set differently.

    The audit for the greedy pass's order dependence, run per row. Partitions
    compare as sets of index sets, so representative choice and cluster order
    cannot produce a false disagreement.
    """
    answer_set = [greedy_answer, *sample_answers]
    greedy = cluster_answers(answer_set, model, spec)
    exhaustive = cluster_answers_exhaustive(answer_set, model, spec)

    def canonical(part: list[list[int]]) -> set[frozenset[int]]:
        return {frozenset(c) for c in part}

    return canonical(greedy) != canonical(exhaustive)


def compute_family_b(
    greedy_answer: str,
    sample_answers: Sequence[str],
    model: EntailmentModel,
    spec: NLISpec,
) -> dict[str, float]:
    """Every family-B signal, in raw units.

    The answer set is `[greedy, *samples]`: the greedy answer is element 0, so
    the clustering measures whether the answer under evaluation sits in the
    majority meaning rather than describing an unrelated sample population.

    `disagreement_rate` is computed against the samples only, since the greedy
    answer trivially agrees with itself and including it would compress the
    signal's range by a factor of (N+1)/N for no information.
    """
    answer_set = [greedy_answer, *sample_answers]
    nan = float("nan")
    if not sample_answers:
        # Nothing to be consistent with. NaN, not 0.0: 0.0 would read as
        # perfect agreement, which is the opposite of "unmeasured".
        return {s.name: nan for s in FAMILY_B_SIGNALS}

    greedy_norm = normalize_answer(greedy_answer)
    disagreements = sum(1 for a in sample_answers if normalize_answer(a) != greedy_norm)
    distinct = count_distinct(answer_set)

    clusters = cluster_answers(answer_set, model, spec)
    entropy = cluster_entropy(clusters)
    # Normalizing by ln(set size) puts the value on [0, 1] so an N=2 and an N=5
    # run are comparable. A set of one has ln(1) = 0 in the denominator and no
    # meaningful normalization, hence NaN.
    max_entropy = math.log(len(answer_set)) if len(answer_set) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else nan

    return {
        DISAGREEMENT_RATE.name: disagreements / len(sample_answers),
        DISTINCT_COUNT.name: float(distinct),
        DISTINCT_FRACTION.name: distinct / len(answer_set),
        MEAN_PAIRWISE_F1.name: mean_pairwise_token_f1(answer_set),
        SEMANTIC_ENTROPY.name: entropy,
        SEMANTIC_ENTROPY_NORM.name: normalized_entropy,
    }


def compute_family_b_samples_only(
    sample_answers: Sequence[str],
    model: EntailmentModel,
    spec: NLISpec,
) -> dict[str, float]:
    """The six family-B statistics over the stochastic samples alone.

    The greedy answer (T=0) is excluded: published semantic-entropy work
    estimates one output distribution at one temperature, and the greedy draw
    is a mode-seeking point mass from a different distribution. `greedy` is
    still the answer being scored downstream; these columns describe the
    distribution it was drawn alongside. `disagreement_rate` is read against
    the plurality sample answer (first-seen wins ties) since there is no
    greedy reference in this variant. Missing samples are NaN, as ever.
    """
    nan = float("nan")
    if not sample_answers:
        return {s.name: nan for s in FAMILY_B_SAMPLES_ONLY_SIGNALS}

    normalized = [normalize_answer(a) for a in sample_answers]
    counts: dict[str, int] = {}
    plurality = normalized[0]
    for text in normalized:
        counts[text] = counts.get(text, 0) + 1
        if counts[text] > counts[plurality]:
            plurality = text
    disagreements = sum(1 for text in normalized if text != plurality)
    distinct = count_distinct(sample_answers)

    clusters = cluster_answers(list(sample_answers), model, spec)
    entropy = cluster_entropy(clusters)
    max_entropy = math.log(len(sample_answers)) if len(sample_answers) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else nan

    return {
        DISAGREEMENT_RATE_SO.name: disagreements / len(sample_answers),
        DISTINCT_COUNT_SO.name: float(distinct),
        DISTINCT_FRACTION_SO.name: distinct / len(sample_answers),
        MEAN_PAIRWISE_F1_SO.name: mean_pairwise_token_f1(list(sample_answers)),
        SEMANTIC_ENTROPY_SO.name: entropy,
        SEMANTIC_ENTROPY_NORM_SO.name: normalized_entropy,
    }
