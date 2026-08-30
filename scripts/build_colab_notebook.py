"""Write `notebooks/run_on_colab.ipynb`.

The notebook is generated rather than hand-edited because a hand-edited .ipynb
is a JSON file with source split across string lists, and a stray comma in it
fails at open time rather than at run time. Generating it means the cell text
lives here as ordinary Python strings and the JSON is always well-formed.

Run this after changing the notebook's text:

    uv run python scripts/build_colab_notebook.py

It is idempotent and writes the same bytes for the same input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "run_on_colab.ipynb"

CELL_1_ESTIMATE = """\
# Run #3 on a free Colab T4

This notebook runs the whole benchmark at **n=600** with a **balanced 300 PopQA /
300 TriviaQA** split against `configs/run3_gpu.yaml`, on a free Colab T4. It is
meant to run top to bottom with no editing.

Run #3 **has not been executed**. Nothing in this repository reports a run #3
number. What is below is a projection.

## Runtime: an estimate, not a measurement

Every figure in this section is derived, and the derivation is shown so it can be
disagreed with. None of it has been measured on a T4.

**Generation.**

- Run #2 measured **16.8-19.0 s/question** on 2 CPU cores with
  Qwen2.5-0.5B-Instruct. That is the only measured anchor here.
- 0.5B to 3B is roughly **6.2x** more compute per token (3.09B / 0.494B
  parameters).
- Autoregressive decode at batch size 1 is memory-bandwidth-bound, not
  FLOP-bound: each token reads the whole weight matrix once. A T4 at
  **~320 GB/s** against CPU DDR4 at **~20 GB/s** is roughly **16x**.
- **No batching speedup is applied.** An earlier version of this estimate
  assumed a 3x speedup from batched generation. That no longer applies:
  `generation_batch_size` is pinned at **1** because open defect **D27**
  (`docs/DECISIONS.md`, session 7) records that padding a ragged batch perturbs
  per-token logprobs by up to 2.52e-02, which is the input signal family A is
  computed from. The defect is unfixed, so the batch stays at 1 and the
  throughput gain it would have bought is not in this arithmetic.

So `16.8-19.0 s/question x 6.2 / 16` gives roughly **6.5-7.4 s/question**, and
600 questions is roughly **65-75 minutes of generation**.

**The other stages.**

- **Family B** adds five sampled generations per question, in the same per-seed
  loop, and then one NLI clustering pass. It is the largest single addition and
  it scales with the generation figure above.
- **Labeling: about 18 minutes.** API-bound, not GPU-bound, so the hardware
  change does not help it. Two judges over the rows exact match does not settle.
- **Analysis: 10-20 minutes at n=600.** Measured from the bootstrap cost, not
  guessed: **4.67 s per AUROC interval** at n=600 against **1.14 s at n=120**, at
  10,000 resamples, over 21 signals plus the per-dataset breakdown
  (`docs/DECISIONS.md` D25).

## Total: roughly 1.5 to 2.5 hours

That is a band, not a point, and it is an estimate.

Two consequences worth reading before you start:

- **It exceeds a single free Colab session's typical idle tolerance.** A free
  session is reclaimed after a period of inactivity and has a wall-clock ceiling.
  Expect at least one disconnect over a run this long.
- **The pipeline is resumable from checkpoints, so a disconnect loses at most the
  current stage.** Each of `build-dataset`, `generate`, `score-signals`, `label`
  and `analyze` writes an atomic checkpoint into `data/run3/` and resumes by
  question id. Re-running this notebook from the top after a disconnect
  re-executes the completed stages as no-ops and picks up where generation
  stopped. It does **not** re-run the judges on rows already labelled, and the
  response cache means it does not re-generate answers it already has.

Environment setup (clone plus dependency install, including a CUDA torch wheel)
is a few minutes on top of the band above and is not counted in it.
"""

CELL_2_CLONE = """\
# Clone the repository and install the pinned dependencies.
#
# Nothing from this project is imported into the notebook kernel. Every stage
# runs as its own subprocess through the `unc-bench` console script, which is
# deliberate: the pinned dependency set includes numpy 1.26.4, Colab ships
# numpy 2.x, and a downgrade underneath an already-imported pandas or matplotlib
# is an ABI problem that surfaces as an unrelated-looking crash. A fresh
# subprocess per stage sidesteps it entirely.

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/urrra39/llm-uncertainty-benchmark.git"
REPO_DIR = "/content/llm-uncertainty-benchmark"
CONFIG = "configs/run3_gpu.yaml"

if not Path(REPO_DIR).is_dir():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
print("cwd:", Path.cwd())
subprocess.run(["git", "log", "-1", "--oneline"], check=True)

# `.[local]` is the project's own optional extra: torch 2.5.1, transformers
# 4.46.3, sentencepiece, protobuf, accelerate. Installing the extra rather than
# hand-listing packages means this notebook cannot drift from pyproject.toml.
#
# This pulls a CUDA torch wheel and takes a few minutes. If Colab already has a
# compatible torch the resolver reuses it.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[local]"],
    check=True,
)
print("install finished")
"""

CELL_3_GPU = """\
# A GPU is a hard requirement, not a preference. Fail here rather than 70
# minutes into a CPU run that would take days.
#
# Runtime > Change runtime type > T4 GPU, then re-run from the top.

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "no CUDA device visible. Set Runtime > Change runtime type > T4 GPU "
        "and re-run this notebook from the first cell."
    )

index = torch.cuda.current_device()
props = torch.cuda.get_device_properties(index)
total_gib = props.total_memory / 1024**3

print(f"device            : {torch.cuda.get_device_name(index)}")
print(f"total memory      : {total_gib:.2f} GiB ({props.total_memory} bytes)")
print(f"compute capability: {props.major}.{props.minor}")
print(f"torch             : {torch.__version__}")
print(f"cuda (torch build): {torch.version.cuda}")

# Qwen2.5-3B-Instruct at fp16 is about 6.2 GB of weights. Under this it will not
# load, and the config's fp16 choice assumes a Turing card with fp16 tensor cores
# and no hardware bf16 (docs/DECISIONS.md D23).
if total_gib < 12:
    print(
        f"\\nWARNING: {total_gib:.1f} GiB is below the ~12 GiB this config assumes. "
        "The 3B model in fp16 plus its KV cache may not fit."
    )
if (props.major, props.minor) != (7, 5):
    print(
        f"\\nNOTE: compute capability {props.major}.{props.minor} is not the T4's 7.5. "
        "float16 is still exact; the runtime estimate in the first cell is not."
    )
"""

CELL_4_KEY = """\
# The judge API key.
#
# `getpass` reads it without echoing it and without writing it anywhere. It goes
# into this kernel's environment only, which the stage subprocesses inherit. It
# is not written to disk, not saved into the notebook, and not printed. Closing
# the runtime discards it.
#
# The key is read from GSK_API_KEY, which is what `configs/run3_gpu.yaml` names
# in `judges.primary.api_key_env`. Only the `label` stage uses it; generation and
# analysis do not call any API.

import getpass
import os

if not os.environ.get("GSK_API_KEY"):
    os.environ["GSK_API_KEY"] = getpass.getpass("judge API key (GSK_API_KEY): ")

key = os.environ.get("GSK_API_KEY", "")
if not key.strip():
    raise RuntimeError("no judge API key given; the label stage cannot run without one")
print(f"key set, length {len(key)} characters (value not shown)")
"""

CELL_5_STAGE_HELPER = '''\
# One helper for every stage, so progress is visible.
#
# The stages print `s/item` and an ETA with flush=True as they go. Piping through
# a captured buffer would hold that output until the stage finished, which for a
# 70-minute generation is the same as printing nothing, so the child's stdout is
# streamed straight through to the notebook.

import subprocess
import sys
import time


def stage(*args: str) -> None:
    """Run one `unc-bench` subcommand against the run #3 config, streaming output."""
    cmd = [sys.executable, "-m", "unc_bench.cli", *args, "--config", CONFIG]
    print(f"\\n=== {' '.join(args)} ===", flush=True)
    started = time.monotonic()
    proc = subprocess.run(cmd)
    elapsed = time.monotonic() - started
    print(f"=== {' '.join(args)} finished in {elapsed / 60:.1f} min ===", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"stage {' '.join(args)} exited {proc.returncode}")


print("helper ready")
'''

CELL_6_BUILD = """\
# Stage 1: draw the question set.
#
# 300 PopQA from 389 candidates after the 8-relation, quantile-0.9 filter, and
# 300 TriviaQA from 4060 after easy_only + min_aliases 20. Both candidate counts
# were measured; see docs/DECISIONS.md D22. Downloads the two source corpora on
# first run.
#
# Idempotent: writes data/run3/dataset.parquet and skips if it already exists.

stage("build-dataset")
"""

CELL_7_GENERATE = """\
# Stage 2: the greedy answer, five sampled continuations, and the verification
# passes, for every question. This is the long one — roughly 65-75 minutes by the
# estimate in the first cell, plus family B's five samples per question.
#
# Resumable by question id. If the session drops, re-run this cell (or the whole
# notebook) and it continues from the last flushed checkpoint rather than
# starting over. Checkpoints flush every 10 questions.

stage("generate")
"""

CELL_8_SIGNALS = """\
# Stage 3: the signal columns, in two passes.
#
# Family B must be its own pass because the DeBERTa NLI model and the generator
# cannot co-reside in the memory the project targets (docs/DECISIONS.md D9). The
# order matters: `actc` needs no model at all, `b` loads DeBERTa.

stage("score-signals", "--family", "actc")
stage("score-signals", "--family", "b")
"""

CELL_9_ABLATION = """\
# The N-ablation: family B AUROC at N = 1, 2, 3, 5, reusing the first N of the
# same five samples so the difference is sample count and not a different draw.
# This is finding 3's evidence.

stage("ablation")
"""

CELL_10_LABEL = """\
# Stage 4: correctness labels. Normalized exact match first, then two LLM judges
# on the rows exact match does not settle, then Cohen's kappa between them.
#
# This is the stage that uses the API key. API-bound, so about 18 minutes by the
# estimate; a GPU does not make it faster. Judge replies are cached, so a re-run
# after a disconnect does not re-pay for rows already judged.

stage("label")
"""

CELL_11_ANALYZE = """\
# Stage 5: the validity gates, the AUROC and AUPRC tables with bootstrap
# intervals, the significance tests, the per-dataset breakdown, and the figures.
#
# Roughly 10-20 minutes at n=600. Run #3 is the first run whose per-dataset block
# carries real bootstrap intervals rather than the analytic approximation the
# README falls back on for run #2 (docs/DECISIONS.md D25).
#
# Writes results_run3.json and figures/run3/*.png.

stage("analyze")
"""

CELL_12_SUMMARY = """\
# The two things worth reading first: whether the run is valid at all, and the
# ranking.
#
# The validity gates come first for the reason they exist. Run #1 produced a full
# 21-signal ranking on 7 positive rows and nothing objected; the gates are code
# in the analysis stage so that cannot happen silently again. If they do not all
# pass, the ranking below is not publishable and the table should be read as a
# record of the failure rather than as a result.

import json
from pathlib import Path

results = json.loads(Path("results_run3.json").read_text(encoding="utf-8"))

gates = results["validity_gates"]
state = "ALL THREE VALIDITY GATES PASS" if gates["all_passed"] else "VALIDITY FAILED"
print(f"{state} - ranking publishable: {gates['ranking_publishable']}")
for gate in gates["gates"]:
    mark = "pass" if gate["passed"] else "FAIL"
    print(f"  [{mark}] {gate['name']}: {gate['observed']}  (requires {gate['requirement']})")
if gates["failed"]:
    print(f"  failed gates: {', '.join(gates['failed'])}")

view = results["views"]["primary"]
catalog = results["signal_catalog"]
print(
    f"\\nn={view['n']}  "
    f"{view['n_incorrect']} incorrect / {view['n_correct']} correct  "
    f"base rate incorrect {view['base_rate_incorrect']:.3f}"
)

print("\\nAUROC, positive class = incorrect answer, 95% bootstrap CI")
print(f"{'signal':<34} {'fam':<4} {'AUROC':>7}  {'95% CI':<18} {'AUPRC':>7}")
print("-" * 76)
for name in view["ranking"]:
    entry = view["signals"][name]
    roc = entry["auroc"]
    prc = entry["auprc"]
    family = catalog.get(name, {}).get("family", "?")
    interval = f"[{roc['ci_low']:.3f}, {roc['ci_high']:.3f}]"
    print(
        f"{name:<34} {family:<4} {roc['point']:>7.3f}  {interval:<18} {prc['point']:>7.3f}"
    )

per_dataset = view.get("per_dataset", {})
if per_dataset.get("available") and per_dataset.get("ci_available"):
    print("\\nPer-dataset AUROC with within-subset bootstrap intervals")
    for source, block in sorted(per_dataset["datasets"].items()):
        print(
            f"\\n  {source}: n={block['n']}, "
            f"{block['n_incorrect']} incorrect / {block['n_correct']} correct"
        )
        signals = block.get("signals", {})
        ordered = sorted(
            (n for n in signals if signals[n].get("auroc") is not None),
            key=lambda n: signals[n]["auroc"],
            reverse=True,
        )
        for name in ordered[:8]:
            item = signals[name]
            ci = item.get("auroc_ci") or {}
            low = ci.get("ci_low")
            high = ci.get("ci_high")
            span = f"[{low:.3f}, {high:.3f}]" if low is not None else "no interval"
            print(f"    {name:<34} {item['auroc']:.3f}  {span}")
"""

CELL_13_ARCHIVE = """\
# One archive with everything a reader needs to check the run, then a download.
#
# Contents: results_run3.json (the numbers), figures/run3/ (the plots, all of
# which are drawn from results_run3.json alone), and data/run3/ (the run
# artifacts: the question set, the generations with their per-token logprobs, the
# signal columns, the labels, the per-row judge verdicts and the timings).
#
# The judge response cache is excluded on purpose. It is request-level scratch
# keyed on model and answer text; judge_verdicts.json is the artifact that
# carries the verdicts in a form the analysis can read.
#
# Committing these artifacts back is the point. Run #2's parquets were lost to
# .gitignore before the policy was fixed, which is why no per-dataset bootstrap
# interval exists for run #2 and why its README table falls back on an analytic
# approximation. Run #3's artifacts are tracked.

import subprocess
from pathlib import Path

ARCHIVE = "run3-artifacts.tar.gz"

members = [p for p in ("results_run3.json", "figures/run3", "data/run3") if Path(p).exists()]
missing = [p for p in ("results_run3.json", "figures/run3", "data/run3") if not Path(p).exists()]
if missing:
    print(f"not present, skipped: {missing}")
if not members:
    raise RuntimeError("nothing to archive; did the analyze stage run?")

subprocess.run(
    ["tar", "czf", ARCHIVE, "--exclude=judge_cache.json", "--exclude=*.tmp", *members],
    check=True,
)
size_mb = Path(ARCHIVE).stat().st_size / 1024**2
print(f"wrote {ARCHIVE}, {size_mb:.2f} MB\\n")

# List what actually went in, rather than asserting it did.
subprocess.run(["tar", "tzf", ARCHIVE], check=True)

# `make check-artifact-size RUN_DIR=data/run3` is the same check the repository
# applies before committing: it fails if any single artifact exceeds 50 MB, at
# which point the archive belongs on a GitHub Release rather than in git. The
# measured worst case for a 600-row generations.parquet was 6.68 MB.
subprocess.run(["make", "check-artifact-size", "RUN_DIR=data/run3"], check=False)

try:
    from google.colab import files  # type: ignore[import-not-found]

    files.download(ARCHIVE)
except Exception as exc:  # not on Colab, or the browser blocked it
    print(f"automatic download unavailable ({exc}); the file is at {Path(ARCHIVE).resolve()}")
"""

CELL_14_COMMIT_BACK = """\
## Committing the artifacts back

Unpack `run3-artifacts.tar.gz` at the root of a clone and commit it. The paths in
the archive are already repository-relative, and `.gitignore` is set up to track
exactly these files (`data/<run>/` outputs are tracked; caches, `judge_cache.json`
and temp files are not).

```bash
tar xzf run3-artifacts.tar.gz
make check-artifact-size RUN_DIR=data/run3
git add results_run3.json figures/run3 data/run3
git commit -m "feat(run3): add the n=600 balanced GPU run's results and artifacts"
git push
```

`make check-artifact-size` fails if any single file exceeds 50 MB. If it does,
use `make export-artifacts RUN_DIR=data/run3` and attach the archive to a GitHub
Release instead of committing the parquets.

Two things to do by hand afterwards, because neither should be automated:

1. **Compare the two runs rather than overwriting run #2.**
   `unc-bench compare-runs results.json results_run3.json --left-label run2
   --right-label run3` prints both AUROC tables side by side with both sets of
   intervals. Run #2's `results.json` is frozen and must not be edited.
2. **Read the validity gate line before quoting any number.** If a gate failed,
   the run is a record of a failure and the ranking is not publishable. That is a
   result too, and it should be written up as one.

## What this notebook does not establish

- The runtime band in the first cell is arithmetic, not a measurement. No stage
  of this pipeline has been timed on a T4.
- Whether the widened PopQA relation set lands inside the pilot gate's 35-65%
  base-rate band is **unverified**. `religion`, `place of birth` and `occupation`
  have never been generated against. Run #2's own 40-row pilot missed its full
  run's base rate by 13 points, so treat the projection with the same suspicion.
- Whether generation is bit-reproducible on a GPU is **unmeasured**, as it is on
  CPU. The `nondeterminism` stage exists and would quantify it; it has not been
  run.
- The 300 PopQA rows are drawn from 389 candidates, a 1.30x margin. The PopQA
  slice is closer to a census of its filter than a sample from a large pool, so
  it is more reproducible and less representative at the same time.
"""


def markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list[str]:
    """Split into the line list nbformat wants: newline kept, except on the last."""
    raw = text.rstrip("\n").split("\n")
    return [line + "\n" for line in raw[:-1]] + [raw[-1]]


def build() -> dict[str, Any]:
    cells = [
        markdown(CELL_1_ESTIMATE),
        code(CELL_2_CLONE),
        code(CELL_3_GPU),
        code(CELL_4_KEY),
        code(CELL_5_STAGE_HELPER),
        code(CELL_6_BUILD),
        code(CELL_7_GENERATE),
        code(CELL_8_SIGNALS),
        code(CELL_9_ABLATION),
        code(CELL_10_LABEL),
        code(CELL_11_ANALYZE),
        code(CELL_12_SUMMARY),
        code(CELL_13_ARCHIVE),
        markdown(CELL_14_COMMIT_BACK),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
