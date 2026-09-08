"""Cross-check every number the documentation quotes against the results files.

The documents — `README.md`, `docs/WITHDRAWN_RUN2.md`,
`docs/LIMITATIONS.md`, `docs/DECISIONS.md`, `docs/AUDIT_RESPONSE.md` and
`data/README.md` — restate runs' numbers in prose. Prose drifts. This script
re-derives each quoted figure from the committed results files and reports a
mismatch, so a disagreement is found by reading the file rather than by
remembering. Three result files: `results_run2b_fixedlabels.json` is primary
(run #2b under the corrected labels); `results_run2b.json` is the archived
pre-fix comparison the README's sensitivity section quotes;
`results_run2_withdrawn.json` is the withdrawn record, checked against the
withdrawn document only.

It is a diagnostic, not a test: it prints and exits non-zero on a mismatch. Run
it by hand after editing any of the documents, and CI runs it on every push:

    uv run python scripts/audit_docs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "AUDIT_RESPONSE.md",
    REPO_ROOT / "docs" / "DECISIONS.md",
    REPO_ROOT / "docs" / "LIMITATIONS.md",
    REPO_ROOT / "docs" / "WITHDRAWN_RUN2.md",
    REPO_ROOT / "data" / "README.md",
)

#: The primary results file. Every ranking README quotes must be traceable here.
#: Run #2b's published record is the FIXED-label analyze (the corrected no-judge
#: rule); the pre-fix set is the Round-11 comparison, pinned separately.
PRIMARY_RESULTS = REPO_ROOT / "results_run2b_fixedlabels.json"
#: The pre-fix run #2b record, quoted only as the labelled sensitivity
#: comparison in the README's Round-11 section.
PREFIX_RESULTS = REPO_ROOT / "results_run2b.json"
#: The withdrawn record. WITHDRAWN_RUN2.md's numbers must be traceable here.
WITHDRAWN_RESULTS = REPO_ROOT / "results_run2_withdrawn.json"


def load() -> dict[str, Any]:
    payload = json.loads(PRIMARY_RESULTS.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def load_withdrawn() -> dict[str, Any]:
    payload = json.loads(WITHDRAWN_RESULTS.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def check_headline(results: dict[str, Any], problems: list[str]) -> None:
    """The counts every document repeats."""
    view = results["views"]["primary"]
    expected = {
        "n": 120,
        "n_correct": 57,
        "n_incorrect": 63,
        "base_rate_incorrect": 0.525,
    }
    for key, want in expected.items():
        got = view[key]
        if got != want:
            problems.append(f"withdrawn views.primary.{key} is {got}, docs assume {want}")

    labels = results["labels"]
    if labels["sources"] != {"exact_match": 54, "judge": 66}:
        problems.append(f"label sources moved: {labels['sources']}")
    if labels["n_abstentions"] != 0:
        problems.append(f"abstentions moved: {labels['n_abstentions']}")
    kappa = labels["kappa"]
    if round(kappa["kappa"], 3) != 0.849:
        problems.append(f"kappa moved: {kappa['kappa']}")
    if kappa["n"] != 66:
        problems.append(f"kappa n moved: {kappa['n']}")
    if results["dataset"]["mix"] != {"popqa": 90, "triviaqa": 30, "simpleqa": 0}:
        problems.append(f"dataset mix moved: {results['dataset']['mix']}")
    if results["frozen_analysis_set"]["primary"]["qid_digest"] != "ffff86216137caed":
        problems.append("qid_digest moved")


#: AUROC and AUPRC point estimates as the README's pooled table prints them.
POOLED = {
    "b_distinct_count": (0.704, 0.724),
    "b_distinct_fraction": (0.704, 0.724),
    "b_disagreement_rate": (0.701, 0.716),
    "b_semantic_entropy": (0.693, 0.712),
    "b_semantic_entropy_normalized": (0.693, 0.712),
    "b_mean_pairwise_f1": (0.691, 0.714),
    "t_question_length": (0.684, 0.697),
    "a_mean_logprob": (0.664, 0.753),
    "a_perplexity": (0.664, 0.753),
    "a_length_normalized_logprob": (0.662, 0.752),
    "a_total_logprob": (0.660, 0.742),
    "c_p_true_plain": (0.659, 0.756),
    "a_min_logprob": (0.649, 0.747),
    "a_max_top5_entropy": (0.636, 0.728),
    "a_mean_top5_entropy": (0.633, 0.689),
    "c_p_true_with_samples": (0.626, 0.713),
    "a_first_token_logprob": (0.580, 0.697),
    "t_answer_length": (0.551, 0.578),
    "a_first_token_margin": (0.545, 0.601),
    "t_random": (0.508, 0.542),
    "c_verbal_confidence": (0.484, 0.527),
}

#: The README's per-dataset primary table, PopQA then TriviaQA then pooled.
PER_DATASET = {
    "b_distinct_count": (0.603, 0.741),
    "b_distinct_fraction": (0.603, 0.741),
    "b_disagreement_rate": (0.601, 0.685),
    "b_semantic_entropy": (0.579, 0.704),
    "b_semantic_entropy_normalized": (0.579, 0.704),
    "b_mean_pairwise_f1": (0.595, 0.691),
    "t_question_length": (0.505, 0.580),
    "a_mean_logprob": (0.514, 0.753),
    "c_p_true_plain": (0.626, 0.660),
    "t_random": (0.511, 0.691),
    "c_verbal_confidence": (0.395, 0.765),
}


def check_pooled(results: dict[str, Any], problems: list[str]) -> None:
    signals = results["views"]["primary"]["signals"]
    for name, (auroc, auprc) in POOLED.items():
        entry = signals.get(name)
        if entry is None:
            problems.append(f"withdrawn doc quotes {name}, which is not in the withdrawn file")
            continue
        got_roc = round(entry["auroc"]["point"], 3)
        got_prc = round(entry["auprc"]["point"], 3)
        if got_roc != auroc:
            problems.append(f"{name} pooled AUROC: results {got_roc}, README {auroc}")
        if got_prc != auprc:
            problems.append(f"{name} pooled AUPRC: results {got_prc}, README {auprc}")


def check_per_dataset(results: dict[str, Any], problems: list[str]) -> None:
    block = results["views"]["primary"]["per_dataset"]
    if not block.get("available"):
        problems.append("per_dataset is unavailable in the withdrawn file")
        return
    for name, (popqa, triviaqa) in PER_DATASET.items():
        for source, want in (("popqa", popqa), ("triviaqa", triviaqa)):
            entry = block["datasets"][source]["signals"].get(name)
            if entry is None:
                problems.append(f"withdrawn doc quotes {name} on {source}, absent from file")
                continue
            got = round(entry["auroc"], 3)
            if got != want:
                problems.append(f"{name} {source} AUROC: results {got}, README {want}")
    popqa = block["datasets"]["popqa"]
    triviaqa = block["datasets"]["triviaqa"]
    if (popqa["n"], triviaqa["n"]) != (90, 30):
        problems.append(f"per-dataset n moved: popqa {popqa['n']}, triviaqa {triviaqa['n']}")
    # The README says PopQA is 40% incorrect and TriviaQA 90%.
    if round(popqa["base_rate_incorrect"], 2) != 0.40:
        problems.append(f"popqa base rate: {popqa['base_rate_incorrect']}")
    if round(triviaqa["base_rate_incorrect"], 2) != 0.90:
        problems.append(f"triviaqa base rate: {triviaqa['base_rate_incorrect']}")
    # And that PopQA has 36 positives against 54 negatives.
    if (popqa["n_incorrect"], popqa["n_correct"]) != (36, 54):
        problems.append(
            f"popqa class counts: {popqa['n_incorrect']} incorrect, {popqa['n_correct']} correct"
        )
    # TriviaQA's 3 correct rows out of 30 is quoted in three documents.
    if (triviaqa["n_incorrect"], triviaqa["n_correct"]) != (27, 3):
        problems.append(
            f"triviaqa class counts: {triviaqa['n_incorrect']} incorrect, "
            f"{triviaqa['n_correct']} correct"
        )
    # Run #3 will change this; run #2's block carries no intervals.
    if block.get("ci_available"):
        problems.append(
            "results.json now has per-dataset intervals, so the README's "
            "Hanley-McNeil fallback is no longer needed"
        )


#: The README's significance table.
SIGNIFICANCE = {
    "a_first_token_logprob": (-0.124, 0.0288, True),
    "t_answer_length": (-0.153, 0.0408, True),
    "a_first_token_margin": (-0.159, 0.0040, True),
    "c_verbal_confidence": (-0.220, 0.0076, True),
    "t_random": (-0.196, 0.0800, False),
    "t_question_length": (-0.020, 1.0000, False),
    "a_mean_logprob": (-0.040, 1.0000, False),
}


def check_significance(results: dict[str, Any], problems: list[str]) -> None:
    sig = results["views"]["primary"]["significance"]
    by_name = {c["name"]: c for c in sig["comparisons"]}
    for name, (delta, p_holm, significant) in SIGNIFICANCE.items():
        row = by_name.get(name)
        if row is None:
            problems.append(f"withdrawn doc quotes a comparison for {name}, absent from file")
            continue
        if round(row["delta_vs_reference"], 3) != delta:
            problems.append(
                f"{name} delta: results {row['delta_vs_reference']:.3f}, README {delta}"
            )
        if round(row["p_value_holm"], 4) != p_holm:
            problems.append(f"{name} p_holm: results {row['p_value_holm']:.4f}, README {p_holm}")
        if bool(row["significant_holm"]) is not significant:
            problems.append(
                f"{name} significance: results {row['significant_holm']}, README {significant}"
            )
    n_significant = sum(1 for c in sig["comparisons"] if c["significant_holm"])
    if n_significant != 4:
        problems.append(f"README says 4 significant comparisons, results has {n_significant}")
    if len(sig["comparisons"]) != 20:
        problems.append(f"README says 20 comparisons, results has {len(sig['comparisons'])}")


#: The README's N-ablation table.
ABLATION = {1: 0.628, 2: 0.678, 3: 0.705, 5: 0.704}


def check_ablation(results: dict[str, Any], problems: list[str]) -> None:
    block = results.get("ablation")
    if not block or "by_n" not in block:
        problems.append("ablation block is unavailable in the withdrawn file")
        return
    by_n = {int(k): v for k, v in block["by_n"].items()}
    for n, want in ABLATION.items():
        entry = by_n.get(n)
        if entry is None:
            problems.append(f"withdrawn doc quotes ablation N={n}, absent from the withdrawn file")
            continue
        got = round(entry["signals"]["b_distinct_count"]["point"], 3)
        if got != want:
            problems.append(f"ablation N={n}: results {got}, README {want}")
    check = block["agreement_with_main_family_b_pass"]
    if check["max_abs_auroc_difference"] != 0.0:
        problems.append(
            f"README says N=5 reproduces family B with max difference 0.000; "
            f"results has {check['max_abs_auroc_difference']}"
        )


#: The README's calibration table: ECE before and after Platt.
CALIBRATION = {
    "c_p_true_plain": (0.200, 0.146),
    "c_p_true_with_samples": (0.330, 0.127),
    "c_verbal_confidence": (0.445, 0.104),
}


def check_calibration(results: dict[str, Any], problems: list[str]) -> None:
    signals = results["views"]["primary"]["signals"]
    for name, (before, after) in CALIBRATION.items():
        entry = signals[name]
        got_before = round(entry["ece_before_platt"], 3)
        got_after = round(entry["ece_after_platt"], 3)
        if got_before != before:
            problems.append(f"{name} ECE before Platt: results {got_before}, README {before}")
        if got_after != after:
            problems.append(f"{name} ECE after Platt: results {got_after}, README {after}")
    if results["analysis_config"]["ece_bins"] != 10:
        problems.append("withdrawn doc says 10 ECE bins; the withdrawn file disagrees")


def check_misc(results: dict[str, Any], problems: list[str]) -> None:
    view = results["views"]["primary"]

    # `docs/OPEN_DEFECTS.md` is generated from OPEN_DEFECTS below. If the table
    # drifted from its source, the defect tracker is decorative.
    generated = render_open_defects()
    defects_path = REPO_ROOT / "docs" / "OPEN_DEFECTS.md"
    if not defects_path.exists():
        problems.append("docs/OPEN_DEFECTS.md is absent; generate it from OPEN_DEFECTS")
    elif defects_path.read_text(encoding="utf-8") != generated:
        problems.append("docs/OPEN_DEFECTS.md drifted from OPEN_DEFECTS in scripts/audit_docs.py")
    view = results["views"]["primary"]

    # The recovered secondary-judge file must carry its minority floor: 53
    # unanimous rows are not a quotable kappa (the D26 trap in miniature).
    recovered_path = REPO_ROOT / "data" / "judge_verdicts_recovered.json"
    if recovered_path.exists():
        recovered = json.loads(recovered_path.read_text(encoding="utf-8"))
        if recovered.get("trustworthy") is not False:
            problems.append("judge_verdicts_recovered.json still claims trustworthy: true")
        if recovered.get("minority_count") != 6:
            problems.append(f"recovered minority count moved: {recovered.get('minority_count')}")
        if recovered.get("kappa_n") != 53:
            problems.append(f"recovered kappa n moved: {recovered.get('kappa_n')}")
    else:
        problems.append("data/judge_verdicts_recovered.json is absent")

    # Length confound: 0.678 in the at-or-below-median stratum, n=91.
    lc = view["length_confound"]
    stratum = lc["strata"]["at_or_below_median_length"]
    if round(stratum["auroc"], 3) != 0.678:
        problems.append(f"length confound stratified AUROC: {stratum['auroc']}")
    if stratum["n"] != 91:
        problems.append(f"length confound stratum n: {stratum['n']}")
    if round(lc["pooled_auroc"], 3) != 0.704:
        problems.append(f"length confound pooled AUROC: {lc['pooled_auroc']}")
    # The README quotes four Spearman-with-length figures.
    corr_len = lc["spearman_with_answer_length"]
    for name, want in (
        ("a_length_normalized_logprob", 0.44),
        ("b_distinct_count", 0.40),
        ("t_question_length", 0.48),
    ):
        got = round(corr_len[name], 2)
        if got != want:
            problems.append(f"{name} Spearman with answer length: results {got}, README {want}")

    # Signal combination: 0.691 CV AUROC against the best single signal's 0.704.
    comb = view["combination"]
    if round(comb["cross_validated_auroc"], 3) != 0.691:
        problems.append(f"combination CV AUROC: {comb['cross_validated_auroc']}")
    if comb["beats_best_single"]:
        problems.append("README says the combination does not beat the best single signal")
    if comb["cv_folds"] != 5:
        problems.append(f"combination folds: {comb['cv_folds']}")
    if comb["signals"] != ["b_distinct_count", "b_disagreement_rate"]:
        problems.append(f"combination signals: {comb['signals']}")

    # Verbalized confidence health: 0 parse failures, 4 distinct values, modal 0.525.
    health = view["verbal_confidence_health"]
    if health["n_parse_failures"] != 0:
        problems.append(f"verbal confidence parse failures: {health['n_parse_failures']}")
    if health["n_distinct_values"] != 4:
        problems.append(f"verbal confidence distinct values: {health['n_distinct_values']}")
    if round(health["modal_share"], 3) != 0.525:
        problems.append(f"verbal confidence modal share: {health['modal_share']}")
    if health["value_distribution"] != {"0.85": 47, "0.89": 1, "0.95": 9, "1.0": 63}:
        problems.append(f"verbal confidence distribution: {health['value_distribution']}")
    if health["effectively_constant"]:
        problems.append("README says verbal confidence is not effectively constant")

    # Cost: family B measured at 6.0x, and the two measured per-item figures.
    cost = results["cost"]
    if round(cost["signals"]["b_distinct_count"]["cost_multiplier"], 1) != 6.0:
        problems.append("family B cost multiplier is not 6.0x")
    if round(cost["measured_family_b_seconds_per_question"], 2) != 0.52:
        problems.append(
            f"family B s/item: {cost['measured_family_b_seconds_per_question']:.3f}, README 0.52"
        )
    if round(cost["measured_generate_seconds_per_question"], 1) != 4.9:
        problems.append(
            f"generate s/item: {cost['measured_generate_seconds_per_question']:.2f}, README 4.9"
        )

    # Validity gates: all three pass, and the observed strings the README quotes.
    gates = {g["name"]: g for g in results["validity_gates"]["gates"]}
    if not results["validity_gates"]["all_passed"]:
        problems.append("withdrawn file says a gate failed; withdrawn doc says all pass")
    if gates["random_baseline_ci_contains_chance"]["observed"] != "AUROC 0.508 [0.404, 0.611]":
        problems.append(
            f"random gate observed: {gates['random_baseline_ci_contains_chance']['observed']}"
        )
    if gates["minimum_rows_per_class"]["observed"] != "63 incorrect, 57 correct":
        problems.append(f"class gate observed: {gates['minimum_rows_per_class']['observed']}")
    if gates["abstention_rate_below_ceiling"]["observed"] != "0/120 = 0.000":
        problems.append(
            f"abstention gate observed: {gates['abstention_rate_below_ceiling']['observed']}"
        )

    # Environment, as the README's hardware section states it.
    env = results["environment"]
    if not env["platform"].startswith("Linux-6.1.155-x86_64"):
        problems.append(f"platform: {env['platform']}")
    if env["python"] != "3.11.16":
        problems.append(f"python: {env['python']}")

    # Seeds, as the README's determinism section lists them.
    seeds = results["seeds"]
    want_seeds = {
        "greedy_seed": 0,
        "sampling_seed_base": 1000,
        "dataset_seed": 12345,
        "split_seed": 20260101,
        "logreg_seed": 31337,
        "bootstrap_seed": 987654321,
    }
    for key, want in want_seeds.items():
        if seeds[key] != want:
            problems.append(f"seed {key}: results {seeds[key]}, README {want}")

    # Correlation: the three rank-equivalent pairs at exactly +1.000.
    corr = view["correlation"]
    matrix = corr["spearman"]
    names = corr["names"]
    for left, right in (
        ("a_mean_logprob", "a_perplexity"),
        ("b_distinct_count", "b_distinct_fraction"),
        ("b_semantic_entropy", "b_semantic_entropy_normalized"),
    ):
        value = matrix[names.index(left)][names.index(right)]
        if round(value, 3) != 1.0:
            problems.append(f"Spearman {left} vs {right}: {value}")


def check_cross_document(results: dict[str, Any], problems: list[str]) -> None:
    """Claims the documents make about each other and about the repository."""
    texts = {path: path.read_text(encoding="utf-8") for path in DOCS}

    # A stale gitignore claim. The policy was inverted: run outputs are tracked.
    data_readme = texts[REPO_ROOT / "data" / "README.md"]
    if "`.gitignore` excludes `data/*.parquet`, `data/artifacts/`" in data_readme:
        problems.append(
            "data/README.md still describes the pre-fix gitignore policy "
            "(claims data/*.parquet is excluded wholesale)"
        )

    # The superseded runtime estimate assumed a 3x batching speedup.
    for path, text in texts.items():
        # "3x" is spelled two ways in the superseded text; the second uses U+00D7.
        multiplication_sign = "\u00d7"
        stale_figures = (
            "55-70 min",
            "55 to 70 min",
            "3x batching speedup",
            f"3{multiplication_sign} batching speedup",
        )
        for stale in stale_figures:
            if stale in text:
                label = str(path.relative_to(REPO_ROOT))
                problems.append(f"{label} contains the superseded runtime figure {stale!r}")

    decisions = texts[REPO_ROOT / "docs" / "DECISIONS.md"]
    if "only a session boundary" not in decisions:
        problems.append(
            "docs/DECISIONS.md lacks the 66-vs-60 reconciliation note "
            "(run #1's 60 vs run #2's 66 must be marked as a session boundary)"
        )
    # The README may not quote a ranking without stating the label-quality
    # status beside it (Part A3): the committed run predates the fourth gate,
    # so the statement is prose rather than a value from results.json.
    readme = texts[REPO_ROOT / "README.md"]
    if "human_label_coverage" not in readme:
        problems.append("README.md quotes rankings without naming the human_label_coverage gate")
    # The two-phase human gates must be named identically in the gate module,
    # the pre-registration and the defect tracker, or the ordering control
    # silently forks into three versions of the rule.
    prereg = (REPO_ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8")
    for gate in ("labeling_protocol_validated", "human_label_coverage"):
        if gate not in prereg:
            problems.append(f"docs/PREREGISTRATION.md does not name the {gate} gate")
    defects_text = (REPO_ROOT / "docs" / "OPEN_DEFECTS.md").read_text(encoding="utf-8")
    for gate in ("labeling_protocol_validated", "human_label_coverage"):
        if gate not in defects_text:
            problems.append(f"docs/OPEN_DEFECTS.md does not name the {gate} gate")
    check_audit_response_structure(problems)

    # The withdrawal bound is arithmetic over committed artifacts, not prose:
    # alias-decided echo rows in the observed 100 plus all 20 unobserved rows.
    echo_path = REPO_ROOT / "data" / "echo_contamination_report.json"
    if echo_path.exists():
        echo = json.loads(echo_path.read_text(encoding="utf-8"))
        bound = int(echo["n_echo_decided_by_subject_in_alias_list"]) + (
            int(results["views"]["primary"]["n"]) - int(echo["n_rows"])
        )
        for doc in ("README.md", "docs/DECISIONS.md", "docs/AUDIT_RESPONSE.md"):
            text = (REPO_ROOT / doc).read_text(encoding="utf-8")
            if str(bound) not in text:
                problems.append(
                    f"{doc} does not state the {bound}-row withdrawal bound "
                    "derived from the echo report"
                )
    else:
        problems.append("data/echo_contamination_report.json is absent")
    # The session-7 status block claimed three files did not exist.
    for claim in (
        "`configs/run3_gpu.yaml` does not exist.",
        "`notebooks/run_on_colab.ipynb` does not exist.",
        "The README has no run #3 section",
        "The four documents have not been audited against each other.",
        "Run #3 is therefore **not yet possible** from this repository.",
    ):
        if claim in decisions:
            problems.append(f"docs/DECISIONS.md still asserts: {claim!r}")

    # Files the documents reference must exist. Three prefixes are exempt: the
    # documents deliberately name paths that are gone or not yet written, and
    # say so in the surrounding prose. `data/run2/` and `data/artifacts/` are
    # run #2's lost artifact directories, whose absence is the subject of the
    # passages that name them. `data/run3/`, `figures/run3/` and
    # `results_run3.json` are run #3's outputs, and run #3 has not been run.
    absent_by_design = (
        "data/run2",
        "data/artifacts",
        "data/run3",
        "figures/run3",
        "data/cache",
        "data/raw",
    )
    for path, text in texts.items():
        label = str(path.relative_to(REPO_ROOT))
        pattern = r"`((?:configs|docs|data|figures|scripts|notebooks)/[\w./*-]+)`"
        for match in re.finditer(pattern, text):
            target = match.group(1)
            if "*" in target or target.startswith(absent_by_design):
                continue
            if not (REPO_ROOT / target).exists():
                problems.append(f"{label} references {target}, which does not exist")

    # Every config named in the documents must load.
    from unc_bench.config import Config

    for path, text in texts.items():
        label = str(path.relative_to(REPO_ROOT))
        for match in re.finditer(r"configs/([\w.-]+\.yaml)", text):
            target = REPO_ROOT / "configs" / match.group(1)
            if not target.exists():
                problems.append(f"{label} names {target.name}, which does not exist")
                continue
            try:
                Config.load(target)
            except Exception as exc:  # reporting the failure is the point
                problems.append(f"{label} names {target.name}, which fails to load: {exc}")

    # Pilot-gate thresholds: the band is part of the config-to-run contract and
    # the documents must not drift from it. Enforced in check_gate_band, below.

    check_gate_band(problems)


#: The shipped pilot-gate band. Every config and the code default agree on
#: error-rate [0.25, 0.65]; the docs must state run bands as 25-65% and must
#: not print the superseded 35-65% figure (R9 refused that claim: the 0.35
#: value never existed in any config or in git history).
GATE_BAND_LOW = 0.25
GATE_BAND_HIGH = 0.65

#: Stale band spellings the documents must not carry. One historical claim is
#: allowed only where it is explicitly marked as the refused/old figure; none
#: of the audited documents currently does that, so the check is a hard ban.
#: The en dash is spelled with an escape to keep the ambiguous codepoint out of
#: the source (RUF001); docs legitimately use either spelling of the hyphen.
_endash = "\u2013"
STALE_BAND_TOKENS = (f"35{_endash}65", "35-65", "35% floor", f"35{_endash}65%")

#: Config -> the results file that config's analyze stage writes. A config is
#: the run's specification, so its results_json is part of the record: if a
#: config's declared output drifts from this table (or a claimed-current
#: results file disappears), the docs that quote that run drift with it.
CONFIG_RESULTS_MAP: dict[str, tuple[str, str]] = {
    # config name -> (results_json, run_name)
    "run2.yaml": ("results_run2_withdrawn.json", "run2_easy_mix"),
    "run2b_clean.yaml": ("results_run2b.json", "run2b_clean"),
    "run2b_fixedlabels.yaml": ("results_run2b_fixedlabels.json", "run2b_fixedlabels"),
    "run3_gpu.yaml": ("results_run3.json", "run3_gpu_balanced"),
}


def check_gate_band(problems: list[str]) -> None:
    """The pilot-gate band is part of every config's contract: all shipped
    configs must agree on the error-rate band, the code default must match,
    and no audited document may state a different band for a run."""
    from unc_bench.config import Config

    bands: dict[str, tuple[float, float]] = {}
    for config_path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        cfg = Config.load(config_path)
        bands[config_path.name] = (
            cfg.pilot_gate.error_rate_low,
            cfg.pilot_gate.error_rate_high,
        )
    if len(set(bands.values())) > 1:
        problems.append(f"pilot-gate bands differ across configs: {bands}")
    if (GATE_BAND_LOW, GATE_BAND_HIGH) not in bands.values():
        problems.append(
            f"no shipped config carries the canonical band "
            f"{GATE_BAND_LOW}-{GATE_BAND_HIGH}; a deliberate change must arrive "
            "with a DECISIONS entry"
        )
    for path in DOCS:
        label = str(path.relative_to(REPO_ROOT))
        text = path.read_text(encoding="utf-8")
        for token in STALE_BAND_TOKENS:
            if token in text:
                problems.append(f"{label} carries the stale gate band {token!r}")

    # Config-to-run mapping: each named config must still declare the results
    # file and run_name the documents quote, and a current/superseded run's
    # results file must exist on disk (run #3's is absent by design).
    absent_by_design = {"run3_gpu.yaml"}
    for name, (results_name, run_name) in CONFIG_RESULTS_MAP.items():
        target = REPO_ROOT / "configs" / name
        if not target.exists():
            problems.append(f"CONFIG_RESULTS_MAP names {name}, which does not exist")
            continue
        cfg = Config.load(target)
        if str(cfg.paths.results_json) != results_name:
            problems.append(
                f"configs/{name} declares results_json {cfg.paths.results_json}, "
                f"pinned {results_name}"
            )
        if cfg.run_name != run_name:
            problems.append(f"configs/{name} declares run_name {cfg.run_name}, pinned {run_name}")
        if name not in absent_by_design and not (REPO_ROOT / results_name).exists():
            problems.append(f"configs/{name} declares {results_name}, which is absent")


#: Open defects as structured data. `docs/OPEN_DEFECTS.md` is rendered from
#: this list by `render_open_defects`, and `check_misc` fails the audit when
#: the committed file drifts from it — so the tracker cannot go stale without
#: the audit saying so. Permanent consequences (run #2's lost artifacts) are
#: recorded as such rather than as open work.
OPEN_DEFECTS: tuple[dict[str, str], ...] = (
    {
        "id": "D27",
        "title": "Batched generation perturbs per-token logprobs (2.52e-02, CPU/bfloat16)",
        "status": "open",
        "measurement_to_close": (
            "`unc-bench nondeterminism` batch_invariance.pass == true on the "
            "run's target device and dtype (T4, fp16 for run #3)"
        ),
        "blocks": "configs/run3_gpu.yaml generation_batch_size (pinned at 1)",
    },
    {
        "id": "D36",
        "title": "No per-config lock file; concurrent generate halved throughput",
        "status": "closed by per-config lock with stale takeover (tested)",
        "measurement_to_close": (
            "second invocation against the same config refuses to start; "
            "covered by a test that launches two runs"
        ),
        "blocks": "nothing further",
    },
    {
        "id": "HUMAN-COVERAGE",
        "title": "No human has verified any label (gates fail at 0.0)",
        "status": "open",
        "measurement_to_close": (
            "PRE-run labeling_protocol_validated at >= 0.50 on "
            "data/human_validation_sample.csv, then POST-run "
            "human_label_coverage at >= 0.80 on the new run's subsample, per "
            "the ordering in docs/PREREGISTRATION.md"
        ),
        "blocks": "validity_gates.all_passed for every future run",
    },
    {
        "id": "RUN2B-LABELSET",
        "title": (
            "Run #2b's published table is computed from the corrected label set "
            "(data/run2b/labels_fixed.parquet -> results_run2b_fixedlabels.json); "
            "the pre-fix containment-rule set is archived as the comparison "
            "(results_run2b.json)"
        ),
        "status": (
            "code sweep closed (relabel committed and primary); human-coverage "
            "half open, tracked by HUMAN-COVERAGE"
        ),
        "measurement_to_close": (
            "the code sweep's six demonstrable errors are corrected in the "
            "published label set; what remains open is the HUMAN-COVERAGE gate: "
            ">= 0.80 human coverage of the run's fixed-label fuzzy-decided rows "
            "(data/fuzzy_decided_rows_fixed.csv, docs/HUMAN_LABELING.md)"
        ),
        "blocks": "nothing further on the code-sweep half; validity_gates.all_passed "
        "for the run still waits on the human gates (HUMAN-COVERAGE)",
    },
    {
        "id": "RUN2-ARTIFACTS",
        "title": "Run #2 per-row artifacts lost to the old ignore policy",
        "status": "permanent",
        "measurement_to_close": "none recoverable; run #3 artifacts are tracked",
        "blocks": "run #2 per-dataset bootstrap intervals (Hanley-McNeil fallback stands)",
    },
    {
        "id": "GEN-DETERMINISM",
        "title": "Generation reproducibility unmeasured on every run",
        "status": "open",
        "measurement_to_close": (
            "`unc-bench nondeterminism` greedy double-run mismatch rate on the " "run's rows"
        ),
        "blocks": "no claim in either direction (README states this)",
    },
)


def render_open_defects() -> str:
    """The open-defect tracker as markdown, generated from OPEN_DEFECTS."""
    lines = [
        "# Open defects",
        "",
        "Generated from `OPEN_DEFECTS` in `scripts/audit_docs.py` — edit the",
        "source, not this file. The docs audit fails when the two disagree.",
        "",
        "| ID | Title | Status | What closes it | What it blocks |",
        "|---|---|---|---|---|",
    ]
    for defect in OPEN_DEFECTS:
        lines.append(
            f"| {defect['id']} | {defect['title']} | {defect['status']} | "
            f"{defect['measurement_to_close']} | {defect['blocks']} |"
        )
    lines.append("")
    return "\n".join(lines)


#: Run #2b's FIXED-label per-dataset table as README prints it: PopQA, TriviaQA.
#: Point estimates only; the README's cells carry the CIs from the same file.
FIXED_PER_DATASET = {
    "b_mean_pairwise_f1": (0.709, 0.701),
    "c_p_true_plain": (0.828, 0.568),
    "b_disagreement_rate": (0.722, 0.660),
    "b_mean_pairwise_f1_samples_only": (0.687, 0.682),
    "b_distinct_count": (0.701, 0.655),
    "a_length_normalized_logprob": (0.747, 0.602),
    "b_disagreement_rate_samples_only": (0.692, 0.647),
    "a_total_logprob": (0.759, 0.573),
    "b_distinct_count_samples_only": (0.683, 0.643),
    "a_min_logprob": (0.706, 0.589),
    "b_semantic_entropy": (0.619, 0.665),
    "a_mean_logprob": (0.698, 0.571),
    "b_semantic_entropy_samples_only": (0.603, 0.663),
    "a_max_top5_entropy": (0.703, 0.543),
    "c_p_true_with_samples": (0.727, 0.505),
    "a_first_token_logprob": (0.582, 0.616),
    "a_mean_top5_entropy": (0.680, 0.511),
    "t_question_length": (0.499, 0.657),
    "t_answer_length": (0.562, 0.503),
    "c_verbal_confidence": (0.504, 0.550),
    "t_random": (0.481, 0.570),
    "a_first_token_margin": (0.580, 0.452),
}

#: The fixed-label stratified sort key the README's primary table is ordered by.
FIXED_STRATIFIED = {
    "b_mean_pairwise_f1": 0.705,
    "c_p_true_plain": 0.699,
    "b_disagreement_rate": 0.691,
    "b_mean_pairwise_f1_samples_only": 0.684,
    "b_distinct_count": 0.678,
    "a_length_normalized_logprob": 0.675,
    "b_disagreement_rate_samples_only": 0.670,
    "a_total_logprob": 0.667,
    "b_distinct_count_samples_only": 0.663,
    "a_min_logprob": 0.648,
    "b_semantic_entropy": 0.642,
    "a_mean_logprob": 0.635,
    "b_semantic_entropy_samples_only": 0.633,
    "a_max_top5_entropy": 0.623,
    "c_p_true_with_samples": 0.617,
    "a_first_token_logprob": 0.599,
    "a_mean_top5_entropy": 0.596,
    "t_question_length": 0.577,
    "t_answer_length": 0.533,
    "c_verbal_confidence": 0.527,
    "t_random": 0.525,
    "a_first_token_margin": 0.517,
}


def check_primary_table(results: dict[str, Any], problems: list[str]) -> None:
    """README's run #2b fixed-label numbers against the primary results file."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    view = results["views"]["primary"]
    if (view["n"], view["n_incorrect"], view["n_correct"]) != (119, 68, 51):
        problems.append("primary class counts moved from 68 incorrect / 51 correct at n=119")
    block = view["per_dataset"]
    for name, (popqa, triviaqa) in FIXED_PER_DATASET.items():
        for source, want in (("popqa", popqa), ("triviaqa", triviaqa)):
            entry = block["datasets"][source]["signals"].get(name)
            if entry is None:
                problems.append(f"README quotes {name} on {source}, absent from primary file")
                continue
            got = round(entry["auroc"], 3)
            if got != want:
                problems.append(f"{name} {source} AUROC: primary file {got}, pinned {want}")
            if f"{want:.3f}" not in readme:
                problems.append(f"README does not print {name} {source} {want:.3f}")
    # The table's sort order: stratified descending, PopQA breaking ties. The
    # rendered order is pinned by tests/test_readme_numbers.py; here we pin the
    # sort key and the band facts the prose states.
    stratified = view["stratified"]["signals"]
    top = max(stratified, key=lambda n: stratified[n]["point"])
    if top != "b_mean_pairwise_f1":
        problems.append(f"primary stratified leader moved: {top}")
    lead = stratified[top]
    if (round(lead["point"], 3), round(lead["ci_low"], 3), round(lead["ci_high"], 3)) != (
        0.705,
        0.601,
        0.802,
    ):
        problems.append("primary stratified leader moved from 0.705 [0.601, 0.802]")
    for name, want in FIXED_STRATIFIED.items():
        got = round(stratified[name]["point"], 3)
        if got != want:
            problems.append(f"stratified {name}: primary file {got}, pinned {want}")
    # Band: every scored signal's interval overlaps the leader's, so the README
    # prints no below-the-line separator.
    band = {
        name
        for name, e in stratified.items()
        if e["ci_low"] <= lead["ci_high"] and e["ci_high"] >= lead["ci_low"]
    }
    if len(band) != len(stratified):
        problems.append(
            f"primary band is {len(band)} of {len(stratified)}; README says all overlap"
        )
    # Significance: paired bootstrap, Holm over 21 distinct, one survivor.
    sig = view["significance"]
    if sig["n_comparisons"] != 21:
        problems.append(f"primary distinct comparisons moved: {sig['n_comparisons']}")
    surviving = sorted(c["name"] for c in sig["comparisons"] if c.get("significant_holm_distinct"))
    if surviving != ["a_first_token_margin"]:
        problems.append(f"primary Holm survivors moved: {surviving}")
    # Ablation, cost, gates (unchanged by the relabel).
    by_n = {int(k): v for k, v in results["ablation"]["by_n"].items()}
    for n, want in ((1, 0.641), (2, 0.700), (3, 0.742), (5, 0.765)):
        got = round(by_n[n]["signals"]["b_distinct_count"]["point"], 3)
        if got != want:
            problems.append(f"primary ablation N={n}: file {got}, pinned {want}")
    cost_b = results["cost"]["signals"]["b_disagreement_rate"]
    if round(cost_b.get("token_multiplier") or 0.0, 2) != 6.01:
        problems.append("primary family-B token multiplier moved from 6.01")
    gates = {g["name"]: g for g in results["validity_gates"]["gates"]}
    if results["validity_gates"]["all_passed"]:
        problems.append("primary gates unexpectedly all pass (human gates should fail)")
    for gate in ("labeling_protocol_validated", "human_label_coverage"):
        if gates[gate]["observed"] != "coverage 0.000":
            problems.append(f"primary {gate} observed moved from coverage 0.000")


def check_prefixed_comparison(problems: list[str]) -> None:
    """The README's Round-11 sensitivity section quotes pre-fix run #2b numbers
    as the object of the labeler measurement. Those must trace to the archived
    pre-fix results file, which is frozen; if it drifts, this fails."""
    if not PREFIX_RESULTS.exists():
        problems.append("results_run2b.json is absent (the pre-fix comparison file)")
        return
    prefixed = json.loads(PREFIX_RESULTS.read_text(encoding="utf-8"))
    view = prefixed["views"]["primary"]
    if (view["n"], view["n_incorrect"], view["n_correct"]) != (120, 71, 49):
        problems.append("pre-fix class counts moved from 71 incorrect / 49 correct at n=120")
    # The README's sensitivity section quotes the pre-fix stratified leader by
    # value; pin it to the frozen pre-fix file.
    stratified = view["stratified"]["signals"]
    got = stratified["b_disagreement_rate"]
    if round(got["point"], 3) != 0.747:
        problems.append("pre-fix b_disagreement_rate stratified moved from 0.747")
    if (round(got["ci_low"], 3), round(got["ci_high"], 3)) != (0.658, 0.830):
        problems.append("pre-fix b_disagreement_rate stratified CI moved from [0.658, 0.830]")
    # The six proven label errors, from the audit file, are the round's anchor.
    audit_path = REPO_ROOT / "data" / "label_error_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        confirmed = sorted(audit.get("confirmed_error_qids", []))
        if len(confirmed) != 6:
            problems.append(f"label-error audit qids moved: {confirmed}")
        bound = audit.get("lower_bound_machine_label_error", {})
        if (bound.get("count"), bound.get("n")) != (6, 120):
            problems.append("label-error lower bound moved from 6/120")
    else:
        problems.append("data/label_error_audit.json is absent")


def check_readme_scope(problems: list[str]) -> None:
    """P0.3: every signal name README prints must exist in the registry;
    defect references must exist in OPEN_DEFECTS; withdrawn-run numbers may
    appear only inside the history section.
    """
    import re as _re

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    import unc_bench.signals.consistency
    import unc_bench.signals.logprob_signals
    import unc_bench.signals.trivial
    import unc_bench.signals.verification  # noqa: F401  (registration side effect)
    from unc_bench.signals.base import registry

    names = set(registry())
    assert names, "signal registry is empty; imports above must register every family"
    for match in _re.finditer(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", readme):
        token = match.group(1)
        if token in names or token in {"qid_digest", "human_label", "machine_label"}:
            continue
        # Config keys and paths share the underscore shape; only fail on
        # tokens that look like signals (family prefix + suffix).
        if token.split("_")[0] in {"a", "b", "c", "t"} and token not in names:
            problems.append(f"README names {token}, absent from the signal registry")

    defect_ids = [d["id"] for d in OPEN_DEFECTS]
    limitations = (REPO_ROOT / "docs" / "LIMITATIONS.md").read_text(encoding="utf-8")
    for doc_label, text in (("README.md", readme), ("LIMITATIONS.md", limitations)):
        for match in _re.finditer(r"\b(HUMAN-COVERAGE|RUN2-ARTIFACTS|GEN-DETERMINISM)\b", text):
            if match.group(1) not in defect_ids:
                problems.append(
                    f"{doc_label} references defect {match.group(1)}, missing from OPEN_DEFECTS"
                )

    # Withdrawn-run fingerprints that occur nowhere in the primary run's
    # tables: the pooled question-length number, the run #1 random baseline,
    # and run #2's exact class counts. Run #2's pooled leader (0.704) and
    # t_question_length (0.684) are deliberately NOT value-fingerprinted:
    # 0.684 is now a legitimate fixed-label stratified point estimate
    # (b_mean_pairwise_f1_samples_only), the R16 collision the structural
    # delta checks exist for. Point estimates that also occur as run #2b
    # interval endpoints (e.g. 0.514) are NOT fingerprinted by value either —
    # deltas against withdrawn baselines are checked structurally below. A
    # bare fingerprint outside history fails; the same number explicitly
    # attributed to run #2 in prose ("in run #2", "withdrawn") is a historical
    # comparison, not a quoted ranking.
    # Stale-scope phrasing fails too: the split moved every table, so
    # references to tables "below"/"beside" and futures that already happened
    # are dangling.
    for stale in (
        "once it has run",
        "beside the tables",
        "The tables stay as the record",
        "The ranking below is publishable",
        "is ready for the hand labelling that opens the gates",
    ):
        if stale in readme:
            problems.append(f"README.md contains stale scope phrasing: {stale!r}")
    # Any README row-count for labelling must equal label-plan's output.
    from unc_bench.stages.label_human import DEFAULT_TARGET, RUN_CSVS, label_plan

    plan = label_plan(RUN_CSVS["run2b"], DEFAULT_TARGET)
    for match in re.finditer(r"label (\d+) rows", readme):
        if int(match.group(1)) != plan["rows_to_label"]:
            problems.append(
                f"README labelling count {match.group(1)} != plan output {plan['rows_to_label']}"
            )
    _check_superlatives(readme, problems)
    history_at = readme.find("## History of withdrawn runs")
    primary_text = readme[:history_at] if history_at >= 0 else readme
    for fingerprint in ("0.746", "63 incorrect / 57 correct"):
        start = 0
        while True:
            at = primary_text.find(fingerprint, start)
            if at < 0:
                break
            context = primary_text[max(0, at - 80) : at + len(fingerprint) + 80]
            if "run #2" not in context and "withdrawn" not in context:
                problems.append(
                    f"withdrawn-run value {fingerprint!r} appears outside the history section"
                )
                break
            start = at + len(fingerprint)
    # Structural delta check: every "X → Y" comparison against a withdrawn
    # baseline must carry the inline marker, so attribution cannot drift off
    # while the numbers stay.
    for match in _re.finditer(r"0\.\d{3}\s*→\s*0\.\d{3}", primary_text):
        if "withdrawn" not in primary_text[max(0, match.start() - 120) : match.end() + 40]:
            problems.append(f"delta {match.group(0)!r} lacks a withdrawn-baseline marker")


def main() -> int:
    results = load()
    withdrawn = load_withdrawn()
    problems: list[str] = []
    # Run #2's record: the withdrawn file against the withdrawn document.
    check_headline(withdrawn, problems)
    check_pooled(withdrawn, problems)
    check_per_dataset(withdrawn, problems)
    check_significance(withdrawn, problems)
    check_ablation(withdrawn, problems)
    check_calibration(withdrawn, problems)
    check_misc(withdrawn, problems)
    check_cross_document(withdrawn, problems)
    # The primary run: README's run #2b fixed-label numbers against the file.
    check_primary_table(results, problems)
    # The pre-fix comparison the README's sensitivity section quotes.
    check_prefixed_comparison(problems)
    check_readme_scope(problems)

    if not problems:
        print("no discrepancies found across the documents and results files")
        return 0
    print(f"{len(problems)} discrepancies:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def _check_superlatives(text: str, problems: list[str]) -> None:
    """A superlative applied to a signal must carry its qualifier.

    Inside the indistinguishable band, "leader"/"best"/"top" performs an
    ordering the statistics do not support — unless the sentence also carries
    the interval, delta or p-value that qualifies it. Interrogatives (the
    title question) are exempt: asking which signal is best is not claiming
    one is.
    """
    head, _, _ = text.partition("## History of withdrawn runs")
    for match in re.finditer(r"\b(leader|best|top)\b", head, flags=re.IGNORECASE):
        window = head[max(0, match.start() - 160) : match.end() + 160]
        if "?" in head[match.end() : match.end() + 60]:
            continue
        # Both hyphen-minus and U+2212 minus: prose uses either. The escape
        # keeps the ambiguous codepoint out of the source (RUF001).
        pat = r"\[[0-9.]+, ?[0-9.]+\]|p_holm|p =|\(-?[0-9.]+,|\(\u2212?[0-9.]+,"
        if re.search(pat, window):
            continue
        problems.append(
            f"unqualified superlative {match.group(1)!r} near: ...{window.strip()[:90]}..."
        )


def check_audit_response_structure(problems: list[str]) -> None:
    """docs/AUDIT_RESPONSE.md must stay navigable: no repeated round heading,
    one settled-refusals section, and no restated refusals outside it."""
    import re as _re

    text = (REPO_ROOT / "docs" / "AUDIT_RESPONSE.md").read_text(encoding="utf-8")
    headings = _re.findall(r"^# Round \d+.*$", text, flags=_re.MULTILINE)
    if len(headings) != len(set(headings)):
        problems.append("docs/AUDIT_RESPONSE.md has a repeated round heading")
    if text.count("## Settled refusals") != 1:
        problems.append("docs/AUDIT_RESPONSE.md must contain exactly one Settled refusals section")
    for round_no in ("7", "8", "9"):
        section = text.split(f"# Round {round_no}:", 1)
        if len(section) < 2:
            continue
        body = section[1].split("# Round ", 1)[0]
        if "## Refused with reason" in body and "See [Settled refusals]" not in body:
            problems.append(
                f"Round {round_no} restates refusals instead of pointing at Settled refusals"
            )


if __name__ == "__main__":
    sys.exit(main())
