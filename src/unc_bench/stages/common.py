"""Shared machinery for the five pipeline stages.

Two things live here, and both exist because of the same constraint: this run
takes about half an hour of CPU time per hundred questions, so an interrupt at
minute twenty must not cost minute one.

Checkpointing. Every stage writes its parquet through `write_checkpoint`, which
writes to a temp file and renames. A stage that dies mid-write leaves the
previous good file in place rather than a truncated one that reads as valid and
silently drops rows.

Resumption by qid. Stages read whatever parquet is already there, ask which qids
it already covers, and skip those. Combined with the response cache this makes
every stage idempotent: rerunning a finished stage is a no-op that issues no
model calls, and rerunning an interrupted one picks up where it stopped.

Nested structures are stored as JSON strings rather than parquet structs. The
per-token logprob payload is a list of dicts of dicts; pyarrow can represent
that, but the schema it infers depends on which rows it happens to see first, so
a chunk where every answer was empty writes a different schema than the next
chunk and the append fails. A JSON string has one schema always.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from unc_bench.config import Config


@dataclass(frozen=True, slots=True)
class StagePaths:
    """Every artifact the pipeline writes, derived from one config."""

    dataset: Path
    generations: Path
    signals_actc: Path
    signals_b: Path
    labels: Path
    pilot_gate: Path
    timings: Path

    @classmethod
    def of(cls, cfg: Config) -> StagePaths:
        root = cfg.paths.artifacts_dir
        return cls(
            dataset=root / "dataset.parquet",
            generations=root / "generations.parquet",
            signals_actc=root / "signals_actc.parquet",
            signals_b=root / "signals_b.parquet",
            labels=root / "labels.parquet",
            pilot_gate=root / "pilot_gate.json",
            timings=root / "timings.json",
        )


def write_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    """Atomic parquet write. A crash mid-write leaves the old file intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_checkpoint(path: Path) -> pd.DataFrame | None:
    """Load a previous checkpoint, or None if there is nothing usable.

    A corrupt file is treated as absent rather than raising. The alternative is
    that one bad byte in a resumable artifact requires deleting it by hand, and
    every row in it is regenerable from the response cache for free.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def done_qids(frame: pd.DataFrame | None) -> set[str]:
    """Which qids a checkpoint already covers."""
    if frame is None or "qid" not in frame.columns:
        return set()
    return {str(q) for q in frame["qid"].tolist()}


def json_str(value: Any) -> str:
    """Compact JSON for a parquet string column."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def json_load(text: str) -> Any:
    """Inverse of `json_str`, tolerant of an empty cell."""
    if not text:
        return None
    return json.loads(text)


class Progress:
    """Throughput reporter for a long serial stage.

    Prints seconds-per-item and a projected finish, because the whole point of
    the pilot is to measure that number rather than guess it. Flushes on every
    line: this runs under `nohup` with output redirected to a file, and Python
    block-buffers a non-tty stdout, so an unflushed writer shows nothing for
    twenty minutes and looks like a hang.
    """

    def __init__(self, label: str, total: int, *, every: int = 1) -> None:
        self.label = label
        self.total = total
        self.every = every
        self.done = 0
        self.started = time.perf_counter()

    def tick(self, note: str = "") -> None:
        self.done += 1
        if self.done % self.every != 0 and self.done != self.total:
            return
        elapsed = time.perf_counter() - self.started
        rate = elapsed / max(self.done, 1)
        remaining = rate * (self.total - self.done)
        print(
            f"[{self.label}] {self.done}/{self.total}  "
            f"{rate:.1f} s/item  elapsed {elapsed / 60:.1f} min  "
            f"eta {remaining / 60:.1f} min  {note}",
            flush=True,
        )

    def summary(self) -> dict[str, float]:
        elapsed = time.perf_counter() - self.started
        return {
            "items": float(self.done),
            "elapsed_s": elapsed,
            "seconds_per_item": elapsed / max(self.done, 1),
        }


def merge_timings(path: Path, stage: str, payload: dict[str, float]) -> None:
    """Accumulate per-stage timings into one JSON file.

    Read-modify-write rather than append, so the file stays valid JSON and a
    rerun of one stage replaces that stage's entry instead of duplicating it.
    """
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            existing = loaded
    existing[stage] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
