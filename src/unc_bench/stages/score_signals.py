"""Stage 3: turn generations into signal columns.

Split into two passes for a memory reason, not a stylistic one.

`run_actc` needs no model at all. Families A, C and T are pure functions of what
stage 2 already persisted, so this pass is fast, cheap, and rerunnable after any
signal-code change without regenerating anything.

`run_b` loads DeBERTa for bidirectional-entailment clustering. It is a separate
entry point because the generator peaks at 1.57 GB of 2.0 GB and the two models
cannot be co-resident (docs/DECISIONS.md D9). By the time this runs, stage 2 has
exited and its memory is back.

Both passes write their own parquet and are resumable at row granularity. The
merge is a left join on qid, so a family-B pass that was interrupted halfway
still produces a usable table with NaN in the B columns for the rest — which is
the correct representation of "not measured", and the analysis reports the
per-signal usable-n rather than silently dropping the column.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from unc_bench.config import Config
from unc_bench.signals.base import orient_all, signal_names
from unc_bench.signals.consistency import (
    FAMILY_B_SAMPLES_ONLY_SIGNALS,
    FAMILY_B_SIGNALS,
    clustering_disagreement,
    compute_family_b,
    compute_family_b_samples_only,
)
from unc_bench.signals.logprob_signals import compute_family_a
from unc_bench.signals.trivial import compute_family_t
from unc_bench.signals.verification import compute_family_c
from unc_bench.stages.common import (
    Progress,
    StagePaths,
    done_qids,
    json_load,
    merge_timings,
    read_checkpoint,
    write_checkpoint,
)
from unc_bench.stages.generate import generation_from_dict, verification_from_dict

FLUSH_EVERY = 25


def run_actc(cfg: Config) -> int:
    """Score families A, C and T. No model loaded.

    Not resumable-by-skipping: the whole pass is a few seconds of arithmetic, so
    it always recomputes from scratch. That is deliberate — it means a fix to a
    signal formula takes effect on rerun instead of being masked by a stale
    checkpoint.
    """
    paths = StagePaths.of(cfg)
    generations = read_checkpoint(paths.generations)
    if generations is None:
        raise FileNotFoundError(f"{paths.generations} is absent; run generate first")

    rows: list[dict[str, Any]] = []
    for record in generations.to_dict(orient="records"):
        qid = str(record["qid"])
        greedy = generation_from_dict(json_load(str(record["greedy"])))
        greedy_answer = str(record.get("greedy_answer") or "")
        question_text = str(record["question"])

        raw: dict[str, float] = {}
        raw.update(compute_family_a(greedy, cfg.signals))
        raw.update(
            compute_family_c(
                verification_from_dict(json_load(str(record["verify_plain"]))),
                verification_from_dict(json_load(str(record["verify_with_samples"]))),
                _optional_float(record.get("verbal_confidence")),
            )
        )
        raw.update(
            compute_family_t(
                question_text,
                greedy_answer,
                qid=qid,
                seed=cfg.dataset_seed,
            )
        )
        out: dict[str, Any] = {"qid": qid}
        out.update(orient_all(raw))
        rows.append(out)

    frame = pd.DataFrame(rows)
    write_checkpoint(frame, paths.signals_actc)
    print(
        f"[score_signals] families A/C/T: {len(frame)} rows, "
        f"{len(frame.columns) - 1} signals -> {paths.signals_actc}",
        flush=True,
    )
    return int(len(frame))


def run_b(cfg: Config, *, limit: int | None = None) -> int:
    """Score family B. Loads the NLI model; run only after the generator exits."""
    paths = StagePaths.of(cfg)
    generations = read_checkpoint(paths.generations)
    if generations is None:
        raise FileNotFoundError(f"{paths.generations} is absent; run generate first")

    previous = read_checkpoint(paths.signals_b)
    already = done_qids(previous)
    records = [r for r in generations.to_dict(orient="records") if str(r["qid"]) not in already]
    if limit is not None:
        records = records[:limit]
    print(
        f"[score_signals] family B: {len(already)} already done, {len(records)} to go",
        flush=True,
    )
    if not records:
        return len(already)

    # Deferred import and construction: this is the 400 MB allocation.
    from unc_bench.signals.nli import DebertaEntailmentModel

    model = DebertaEntailmentModel(cfg.nli)
    print(
        f"[score_signals] NLI id2label={model.id2label}, "
        f"entailment index {model.entailment_index}",
        flush=True,
    )

    rows: list[dict[str, Any]] = (
        [] if previous is None else list(previous.to_dict(orient="records"))  # type: ignore[arg-type]
    )
    progress = Progress("family_b", len(records), every=10)
    errors = 0
    clustering_disagreements = 0
    clustering_audited = 0
    for record in records:
        qid = str(record["qid"])
        try:
            sample_answers = [str(a) for a in (json_load(str(record["sample_answers"])) or [])]
            greedy_answer = str(record.get("greedy_answer") or "")
            raw = compute_family_b(
                greedy_answer,
                sample_answers,
                model,
                cfg.nli,
                clusterer=cfg.nli.primary_clusterer,
            )
            raw.update(
                compute_family_b_samples_only(
                    sample_answers, model, cfg.nli, clusterer=cfg.nli.primary_clusterer
                )
            )
            if _audit_row(qid, cfg):
                clustering_audited += 1
                try:
                    if clustering_disagreement(greedy_answer, sample_answers, model, cfg.nli):
                        clustering_disagreements += 1
                except Exception as audit_exc:
                    print(f"[score_signals] family B audit {qid} failed: {audit_exc}", flush=True)
            out: dict[str, Any] = {"qid": qid}
            out.update(orient_all(raw))
            rows.append(out)
        except Exception as exc:
            errors += 1
            print(f"[score_signals] family B {qid} failed: {exc}", flush=True)
        if len(rows) % FLUSH_EVERY == 0:
            write_checkpoint(pd.DataFrame(rows), paths.signals_b)
        progress.tick(qid)

    write_checkpoint(pd.DataFrame(rows), paths.signals_b)
    timing = progress.summary()
    timing["failures"] = float(errors)
    merge_timings(paths.timings, "family_b", timing)
    clustering_meta = {
        "n_rows_audited": clustering_audited,
        "greedy_vs_exhaustive_disagreements": clustering_disagreements,
        "primary_clusterer": cfg.nli.primary_clusterer,
        "audit_fraction": cfg.nli.audit_fraction,
        "note": (
            "rows where greedy single-pass assignment and transitive-closure "
            "clustering partition the answer set differently; on disagreement "
            "the exhaustive partition is primary"
        ),
    }
    meta_path = cfg.paths.artifacts_dir / "family_b_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(clustering_meta, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[score_signals] family B: {len(rows)} rows -> {paths.signals_b} "
        f"({timing['seconds_per_item']:.2f} s/item, {errors} failures)",
        flush=True,
    )
    return len(rows)


def _audit_row(qid: str, cfg: Config) -> bool:
    """Whether this row gets both clusterers, deterministically per qid.

    A seeded fraction (default 0.2) keeps the audit's NLI cost bounded on weak
    hardware; `force_full_audit` covers validation runs. Deterministic so
    reruns audit the same rows.
    """
    import hashlib

    if cfg.nli.force_full_audit:
        return True
    if cfg.nli.audit_fraction >= 1.0:
        return True
    if cfg.nli.audit_fraction <= 0.0:
        return False
    digest = hashlib.sha256(qid.encode()).digest()
    return (int.from_bytes(digest[:8], "big") / 2**64) < cfg.nli.audit_fraction


def merged_signals(cfg: Config) -> pd.DataFrame:
    """A/C/T joined with B on qid, with missing B columns present as NaN.

    Left join from A/C/T, because that pass covers every generated row while B
    may be partial. Absent B columns are materialized as NaN rather than dropped
    so the results table has a fixed shape and a partially-scored family shows up
    as a reduced usable-n instead of a missing row.
    """
    paths = StagePaths.of(cfg)
    actc = read_checkpoint(paths.signals_actc)
    if actc is None:
        raise FileNotFoundError(f"{paths.signals_actc} is absent; run score-signals first")
    b_frame = read_checkpoint(paths.signals_b)
    if b_frame is None:
        merged = actc.copy()
    else:
        merged = actc.merge(b_frame, on="qid", how="left", validate="one_to_one")
    for spec in (*FAMILY_B_SIGNALS, *FAMILY_B_SAMPLES_ONLY_SIGNALS):
        if spec.name not in merged.columns:
            merged[spec.name] = float("nan")
    # Fixed column order: qid first, then every registered signal in
    # registration order. Column order in the parquet should not depend on which
    # stage happened to write which column first.
    ordered = ["qid", *[n for n in signal_names() if n in merged.columns]]
    return merged.loc[:, ordered]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out
