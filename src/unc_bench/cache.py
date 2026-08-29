"""Content-addressed disk cache for model responses.

The key is a SHA-256 over every input that could change the output: model name,
backend, the fully rendered prompt, and all sampling parameters including the
seed. Two consequences that matter.

First, a rerun of a completed stage costs zero model calls, which is what makes
the five stages resumable after an interrupt.

Second, editing a prompt template or bumping a temperature produces different
keys rather than reusing stale responses under a new configuration. That is the
behaviour you want; the alternative silently mixes prompt versions inside one
results table.

Entries are gzipped JSON sharded two levels deep by key prefix, because a flat
directory with tens of thousands of files is slow to list on most filesystems.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

# Bump when the on-disk entry format changes incompatibly.
CACHE_FORMAT_VERSION = 1

# Fields that participate in the key, in a fixed order. Anything not listed here
# is deliberately excluded because it cannot change the model's output.
KEYED_FIELDS: tuple[str, ...] = (
    "backend",
    "model",
    "prompt",
    "temperature",
    "top_p",
    "seed",
    "max_new_tokens",
    "top_logprobs",
    "n",
    "kind",
)


def cache_key(
    *,
    backend: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    seed: int,
    max_new_tokens: int,
    top_logprobs: int = 0,
    n: int = 1,
    kind: str = "chat",
) -> str:
    """Stable hash of everything that determines a response.

    Floats are formatted with `repr` so 0.7 and 0.70 hash identically while
    0.7 and 0.71 do not.
    """
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "backend": backend,
        "model": model,
        "prompt": prompt,
        "temperature": repr(float(temperature)),
        "top_p": repr(float(top_p)),
        "seed": int(seed),
        "max_new_tokens": int(max_new_tokens),
        "top_logprobs": int(top_logprobs),
        "n": int(n),
        "kind": kind,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """Gzipped-JSON cache on disk. Not thread-safe by design; stages are serial."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def _path(self, key: str) -> Path:
        # Two-level sharding: 256 x 256 possible directories.
        return self.root / key[:2] / key[2:4] / f"{key}.json.gz"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, json.JSONDecodeError, EOFError, zlib.error, UnicodeDecodeError):
            # A truncated or corrupt entry is a miss, not a crash. zlib.error is
            # listed explicitly: it does not inherit from OSError, so an earlier
            # version of this handler let a half-written gzip member propagate.
            self.misses += 1
            return None
        if not isinstance(entry, dict) or entry.get("_v") != CACHE_FORMAT_VERSION:
            self.misses += 1
            return None
        self.hits += 1
        value = entry.get("value")
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"_v": CACHE_FORMAT_VERSION, "key": key, "value": value}
        # Write to a temp file and rename, so an interrupt cannot leave a
        # half-written entry that later reads as valid.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(entry, fh, ensure_ascii=False)
        tmp.replace(path)
        self.writes += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}

    def __len__(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for _ in self.root.rglob("*.json.gz"))
