"""Every numeric claim in README's run #2b sections traces to
results_run2b_fixedlabels.json (the published, fixed-label record).

Two layers. First, the per-dataset table's rendered order must equal the
stated selection rule (all distinct signals, stratified AUROC descending,
PopQA AUROC breaking ties) computed from the file — ad-hoc curation of a
primary table is the defect this pins. Second, every table cell and headline
number must occur in the file to the printed precision. Prose roundings and
conceptual constants (0.50) are not claims and are not checked here;
everything else is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

PRIMARY_NAME = "results_run2b_fixedlabels.json"


def _primary() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((REPO / PRIMARY_NAME).read_text(encoding="utf-8"))
    return payload


def _readme_primary() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    head, _, _ = text.partition("## History of withdrawn runs")
    return head


def _stratified_rows() -> list[tuple[str, float, float, float, float]]:
    """(name, popqa, triviaqa, stratified point, stratified CI) for every
    distinct signal, in the README's stated order."""
    payload = _primary()
    block = payload["views"]["primary"]["per_dataset"]["datasets"]
    stratified = payload["views"]["primary"]["stratified"]["signals"]
    dropped = set(payload["views"]["primary"]["significance"].get("rank_equivalent_dropped", []))
    rows = []
    names = set(block["popqa"]["signals"]) | set(block["triviaqa"]["signals"])
    for name in sorted(names - dropped):
        p = block["popqa"]["signals"][name]["auroc"]
        t = block["triviaqa"]["signals"][name]["auroc"]
        s = stratified[name]
        rows.append((name, round(p, 3), round(t, 3), round(s["point"], 3), s["ci_high"]))
    rows.sort(key=lambda r: (-r[3], -r[1], r[0]))
    return rows


def test_rendered_order_equals_the_stated_selection_rule() -> None:
    text = _readme_primary()
    start = text.find("### Run #2b per-dataset AUROC")
    table = text[start:].split("### Supporting")[0]
    rendered = re.findall(r"\| `([a-z0-9_]+)` \|", table)
    expected = [name for name, _, _, _, _ in _stratified_rows()]
    assert rendered == expected, (
        "rendered table order diverged from stratified-descending: "
        f"{[n for n, m in zip(rendered, expected, strict=True) if n != m][:3]}"
    )
    assert len(rendered) == 22


def _number_universe() -> set[str]:
    payload = _primary()
    view = payload["views"]["primary"]
    block = view["per_dataset"]["datasets"]
    universe: set[str] = set()
    for dataset in ("popqa", "triviaqa"):
        for entry in block[dataset]["signals"].values():
            universe.add(f"{entry['auroc']:.3f}")
            universe.add(f"{entry['auroc_ci']['ci_low']:.3f}")
            universe.add(f"{entry['auroc_ci']['ci_high']:.3f}")
    for entry in view["stratified"]["signals"].values():
        universe.add(f"{entry['point']:.3f}")
        universe.add(f"{entry['ci_low']:.3f}")
        universe.add(f"{entry['ci_high']:.3f}")
    for entry in view["signals"].values():
        universe.add(f"{entry['auroc']['point']:.3f}")
        universe.add(f"{entry['auroc']['ci_low']:.3f}")
        universe.add(f"{entry['auroc']['ci_high']:.3f}")
        universe.add(f"{entry['auprc']['point']:.3f}")
    return universe


def test_every_table_cell_is_in_the_file() -> None:
    universe = _number_universe()
    text = _readme_primary()
    start = text.find("### Run #2b per-dataset AUROC")
    end = text.find("## The significance verdict", start)
    assert start >= 0 and end > start
    table = text[start:end]
    cells = re.findall(r"\b(0\.\d{3})\b", table)
    assert cells, "no table cells found"
    missing = sorted(set(cells) - universe)
    assert not missing, f"table values absent from {PRIMARY_NAME}: {missing}"


def test_headline_numbers_are_in_the_file() -> None:
    payload = _primary()
    view = payload["views"]["primary"]
    universe: set[str] = set()
    for entry in view["signals"].values():
        universe.add(f"{entry['auroc']['point']:.3f}")
        universe.add(f"{entry['auroc']['ci_low']:.3f}")
        universe.add(f"{entry['auroc']['ci_high']:.3f}")
    for level in payload["ablation"]["by_n"].values():
        universe.add(f"{level['signals']['b_distinct_count']['point']:.3f}")
    for comparison in view["significance"]["comparisons"]:
        universe.add(f"{abs(comparison['delta_vs_reference']):.3f}")
        universe.add(f"{comparison['p_value_holm_distinct']:.3f}")
    universe.add(f"{payload['cost']['signals']['b_disagreement_rate']['token_multiplier']:.2f}")
    universe.add(f"{view['verdict']['gap']:.3f}")
    text = _readme_primary()
    # Headline claims that appear both in the README and in the results file.
    # (Table cells, including the stratified leader's CI bounds and the PopQA
    # column leader, are covered by test_every_table_cell_is_in_the_file.)
    for claim in ("0.705", "0.742", "0.739", "0.765", "0.130", "0.046", "6.01"):
        assert claim in text, f"headline claim {claim} missing from README"
        assert claim in universe, f"headline claim {claim} absent from {PRIMARY_NAME}"


def test_every_signal_is_in_one_band_and_no_separator_is_drawn() -> None:
    """Under the fixed labels every scored signal's stratified interval overlaps
    the leader's, so the README prints no below-the-line separator. The band
    count and the "no separator" fact are computed from the file, not asserted
    by hand."""
    payload = _primary()
    signals = payload["views"]["primary"]["stratified"]["signals"]
    lead = max(signals.items(), key=lambda kv: kv[1]["point"])[1]
    assert (round(lead["point"], 3), round(lead["ci_low"], 3), round(lead["ci_high"], 3)) == (
        0.705,
        0.601,
        0.802,
    )
    band = {
        name
        for name, entry in signals.items()
        if entry["ci_low"] <= lead["ci_high"] and entry["ci_high"] >= lead["ci_low"]
    }
    assert len(band) == len(signals) == 27
    dropped = set(payload["views"]["primary"]["significance"].get("rank_equivalent_dropped", []))
    assert len(band - dropped) == 22
    text = _readme_primary()
    assert "27 of 27" in text
    assert "22 distinct" in text
    assert "below this line: stratified interval entirely below the leader's" not in text


def test_tension_paragraph_matches_generated() -> None:
    """A1: the null-vs-rejections paragraph is generated from the file."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    generated = subprocess.run(
        [sys.executable, "scripts/render_tension_paragraph.py"],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    assert generated.returncode == 0
    text = (repo / "README.md").read_text(encoding="utf-8")
    start = text.find("<!-- TENSION:BEGIN -->")
    end = text.find("<!-- TENSION:END -->")
    assert start >= 0 and end > start
    embedded = text[start : end + len("<!-- TENSION:END -->")] + "\n"
    assert embedded == ("<!-- TENSION:BEGIN -->\n" + generated.stdout + "<!-- TENSION:END -->\n")
