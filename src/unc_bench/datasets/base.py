"""Shared machinery for dataset builders.

Each builder downloads one source, extracts (question, gold aliases), and hands
back a deterministic sample. Determinism comes from a seeded numpy Generator
applied to a stably sorted candidate list, so the same seed and the same source
file always give the same questions in the same order. Sampling from an
unsorted pandas frame would depend on row order in the download and quietly
change the benchmark between runs.

Downloads are cached to `paths.raw_dir` so a rerun is offline.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from unc_bench.types import Question


class OfflineError(RuntimeError):
    """Raised when a download is needed but the environment forbids network."""


def offline_mode() -> bool:
    """CI sets UNC_BENCH_OFFLINE=1. Nothing in CI should hit the network."""
    return os.environ.get("UNC_BENCH_OFFLINE", "") == "1"


def cached_download(url: str, dest: Path) -> Path:
    """Fetch `url` to `dest` once. Later calls reuse the file on disk."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline_mode():
        raise OfflineError(f"{dest} is absent and UNC_BENCH_OFFLINE=1 forbids fetching {url}")
    # Deferred import; only reached on a real download.
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as fh:
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


class DatasetBuilder(ABC):
    """One source of short-form factual questions."""

    #: Value written to `Question.dataset`.
    name: str

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)

    @abstractmethod
    def load_candidates(self) -> list[Question]:
        """Every usable item in the source, before sampling."""

    def build(self, n: int, seed: int) -> list[Question]:
        """Deterministic sample of `n` questions.

        Candidates are sorted by qid before sampling so the result does not
        depend on the row order of the downloaded file.
        """
        if n <= 0:
            return []
        candidates = self.load_candidates()
        candidates = sorted(candidates, key=lambda q: q.qid)
        if len(candidates) < n:
            raise ValueError(
                f"{self.name}: asked for {n} questions but only {len(candidates)} are usable"
            )
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(candidates), size=n, replace=False)
        # Sort the chosen indices so the output order is also stable.
        return [candidates[int(i)] for i in sorted(picked)]


def questions_to_frame(questions: list[Question]) -> pd.DataFrame:
    """Typed frame for the parquet artifact."""
    frame = pd.DataFrame(
        {
            "qid": pd.Series([q.qid for q in questions], dtype="string"),
            "dataset": pd.Series([q.dataset for q in questions], dtype="string"),
            "question": pd.Series([q.question for q in questions], dtype="string"),
            # Aliases are a list per row; object dtype is what parquet wants.
            "gold_answers": pd.Series([list(q.gold_answers) for q in questions], dtype="object"),
        }
    )
    if frame["qid"].duplicated().any():
        dupes = frame.loc[frame["qid"].duplicated(), "qid"].tolist()
        raise ValueError(f"duplicate qids across datasets: {dupes[:5]}")
    return frame


def frame_to_questions(frame: pd.DataFrame) -> list[Question]:
    """Inverse of `questions_to_frame`.

    Columns are pulled out as plain Python lists rather than iterating rows.
    pandas-stubs types an itertuples field as a wide scalar union, so indexing
    into the alias list there does not type-check under strict mypy.
    """
    qids = frame["qid"].astype(str).tolist()
    datasets = frame["dataset"].astype(str).tolist()
    questions = frame["question"].astype(str).tolist()
    aliases = frame["gold_answers"].tolist()
    out: list[Question] = []
    for qid, dataset, question, alias_list in zip(qids, datasets, questions, aliases, strict=True):
        out.append(
            Question(
                qid=qid,
                dataset=dataset,
                question=question,
                gold_answers=tuple(str(a) for a in alias_list),
            )
        )
    return out


def clean_alias_list(aliases: list[str]) -> tuple[str, ...]:
    """Drop blanks and exact duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for alias in aliases:
        text = str(alias).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)
