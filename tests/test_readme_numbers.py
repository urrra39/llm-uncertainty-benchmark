"""Every numeric claim in README's run #2b sections traces to results_run2b.json.

Two layers. First, the per-dataset table's rendered order must equal the
stated selection rule (all distinct signals, stratified AUROC descending,
PopQA AUROC breaking ties) computed from the file — ad-hoc curation of a
primary table is the defect this pins. Second, every table cell and headline
number must occur in the file to the printed precision. Prose roundings
("0.02-0.04") and conceptual constants (0.50) are not claims and are not
checked here; everything else is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _primary() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((REPO / "results_run2b.json").read_text(encoding="utf-8"))
    return payload


def _readme_primary() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    head, _, _ = text.partition("## History of withdrawn runs")
    return head


def _table_rows() -> list[tuple[str, float, float, float]]:
    payload = _primary()
    block = payload["views"]["primary"]["per_dataset"]["datasets"]
    names = set(block["popqa"]["signals"]) | set(block["triviaqa"]["signals"])
    dropped = set(payload["views"]["primary"]["significance"].get("rank_equivalent_dropped", []))
    rows = []
    for name in sorted(names - dropped):
        p = block["popqa"]["signals"][name]["auroc"]
        t = block["triviaqa"]["signals"][name]["auroc"]
        rows.append((name, round(p, 3), round(t, 3), round((p + t) / 2, 3)))
    rows.sort(key=lambda r: (-r[3], -r[1], r[0]))
    return rows


def test_rendered_order_equals_the_stated_selection_rule() -> None:
    text = _readme_primary()
    start = text.find("### Run #2b per-dataset AUROC")
    table = text[start:].split("```")[0]
    rendered = re.findall(r"\| `([a-z0-9_]+)` \|", table)
    expected = [name for name, _, _, _ in _table_rows()]
    assert rendered == expected, (
        "rendered table order diverged from stratified-descending: "
        f"{[n for n, m in zip(rendered, expected, strict=True) if n != m][:3]}"
    )
    assert len(rendered) == 22


def test_every_table_cell_is_in_the_file() -> None:
    payload = _primary()
    block = payload["views"]["primary"]["per_dataset"]["datasets"]
    universe: set[str] = set()
    for dataset in ("popqa", "triviaqa"):
        for entry in block[dataset]["signals"].values():
            universe.add(f"{entry['auroc']:.3f}")
            universe.add(f"{entry['auroc_ci']['ci_low']:.3f}")
            universe.add(f"{entry['auroc_ci']['ci_high']:.3f}")
    for entry in payload["views"]["primary"]["stratified"]["signals"].values():
        universe.add(f"{entry['point']:.3f}")
        universe.add(f"{entry['ci_low']:.3f}")
        universe.add(f"{entry['ci_high']:.3f}")
    for _, _popqa, _trivia, s in _table_rows():
        universe.add(f"{s:.3f}")
    text = _readme_primary()
    start = text.find("### Run #2b per-dataset AUROC")
    end = text.find("### What decontamination", start)
    table = text[start:end]
    cells = re.findall(r"\b(0\.\d{3})\b", table)
    assert cells, "no table cells found"
    missing = sorted(set(cells) - universe)
    assert not missing, f"table values absent from results_run2b.json: {missing}"


def test_headline_numbers_are_in_the_file() -> None:
    payload = _primary()
    view = payload["views"]["primary"]
    universe: set[str] = set()
    for entry in view["signals"].values():
        universe.add(f"{entry['auroc']['point']:.3f}")
    for level in payload["ablation"]["by_n"].values():
        universe.add(f"{level['signals']['b_distinct_count']['point']:.3f}")
    for difference in payload["ablation"].get("level_differences", {}).get("comparisons", []):
        if difference["delta_vs_reference"] is None:
            continue
        universe.add(f"{abs(difference['delta_vs_reference']):.3f}")
        if difference["p_value_holm"] is not None:
            universe.add(f"{difference['p_value_holm']:.4f}")
            universe.add(f"{difference['p_value_holm']:.3f}")
    for comparison in view["significance"]["comparisons"]:
        universe.add(f"{abs(comparison['delta_vs_reference']):.3f}")
        universe.add(f"{comparison['p_value_holm_distinct']:.4f}")
        universe.add(f"{comparison['p_value_holm_distinct']:.3f}")
    universe.add(f"{payload['cost']['signals']['b_disagreement_rate']['token_multiplier']:.2f}")
    universe.add(f"{view['verdict']['gap']:.3f}")
    text = _readme_primary()
    for claim in ("0.799", "0.008", "0.276", "0.0076", "0.124", "0.016", "6.01"):
        assert claim in text, f"headline claim {claim} missing from README"
        assert claim in universe, f"headline claim {claim} absent from results_run2b.json"
