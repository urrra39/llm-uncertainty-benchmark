"""The artifact-tracking policy, asserted against the real .gitignore.

Run #2's per-row scores were lost because `.gitignore` excluded the directory
they were written to, and nothing in the repository objected. The policy is now
inverted — run outputs are tracked, caches and scratch are not — and these tests
exist so a future edit to `.gitignore` that re-excludes an artifact fails here
rather than silently costing another run's data.

`git check-ignore` is the authority, not a re-implementation of gitignore
pattern semantics. Re-implementing them would test my reading of the manual
rather than git's behaviour, and the subtlety that caused the original loss —
a directory-wide pattern shadowing a narrower one — lives exactly in those
semantics.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: One run's outputs. Every one of these must be committable.
TRACKED_ARTIFACTS = (
    "dataset.parquet",
    "dataset_meta.json",
    "generations.parquet",
    "signals_actc.parquet",
    "signals_b.parquet",
    "labels.parquet",
    "judge_verdicts.json",
    "ablation.json",
    "nondeterminism.json",
    "timings.json",
    "pilot_gate.json",
)

#: Caches and scratch. Every one of these must stay ignored.
IGNORED_SCRATCH = (
    "judge_cache.json",
    "generations.parquet.tmp",
)


def _git_available() -> bool:
    return shutil.which("git") is not None and (REPO / ".git").exists()


needs_git = pytest.mark.skipif(
    not _git_available(),
    reason="needs a git checkout; check-ignore is the authority on these patterns",
)


def _is_ignored(relative: str) -> bool:
    """Ask git, on a path that need not exist.

    `check-ignore --no-index` answers from the ignore rules alone, so no file has
    to be created inside the working tree and the test leaves nothing behind.
    Exit 0 means ignored, 1 means not ignored, anything else is a real failure.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore failed on {relative!r}: "
            f"rc={result.returncode} {result.stderr.decode(errors='replace')}"
        )
    return result.returncode == 0


@needs_git
@pytest.mark.parametrize("artifact", TRACKED_ARTIFACTS)
@pytest.mark.parametrize("run_name", ["run3", "run4_gpu", "some_future_run"])
def test_run_artifacts_are_trackable(run_name: str, artifact: str) -> None:
    """`data/<run_name>/<artifact>` must not be ignored, for any run name.

    Parameterized over several run names because the loss happened under one
    specific directory (`data/run2/`) and a rule written for that name only
    would leave the next run just as exposed.
    """
    path = f"data/{run_name}/{artifact}"
    assert not _is_ignored(path), (
        f"{path} is ignored. Run #2's artifacts were lost exactly this way; "
        "see docs/DECISIONS.md, 'Run artifacts are tracked from run #3 on'."
    )


@needs_git
@pytest.mark.parametrize("scratch", IGNORED_SCRATCH)
@pytest.mark.parametrize("run_name", ["run3", "run4_gpu"])
def test_scratch_stays_ignored(run_name: str, scratch: str) -> None:
    """Caches and temp files must stay out of the repository."""
    path = f"data/{run_name}/{scratch}"
    assert _is_ignored(path), f"{path} should be ignored but is not"


@needs_git
@pytest.mark.parametrize(
    "path",
    [
        "data/cache/deadbeef.json",
        "data/raw/popqa_test.tsv",
        "data/raw/triviaqa_rc_nocontext_validation.parquet",
        "dist/run3-artifacts.tar.gz",
        "hf_home/models--Qwen--Qwen2.5-3B-Instruct/blob",
        "data/scratch.parquet",
    ],
)
def test_caches_and_downloads_stay_ignored(path: str) -> None:
    """Regenerable bulk stays out. These are large and no result depends on them."""
    assert _is_ignored(path), f"{path} should be ignored but is not"


@needs_git
@pytest.mark.parametrize(
    "path",
    ["results.json", "figures/auroc.png", "data/README.md", "configs/run3_gpu.yaml"],
)
def test_published_outputs_are_not_ignored(path: str) -> None:
    """The project's own output stays committed. This was already true; pin it."""
    assert not _is_ignored(path), f"{path} must be tracked but is ignored"


@needs_git
def test_committed_run2_pilot_dataset_is_still_tracked() -> None:
    """The one committed run artifact must survive the policy change.

    `data/run2_pilot/dataset.parquet` is cited by docs/DECISIONS.md for its
    measured base rates. A rule that ignored it would not remove the file, but it
    would stop the next `git add` from noticing a change to it.
    """
    assert not _is_ignored("data/run2_pilot/dataset.parquet")
    assert (REPO / "data" / "run2_pilot" / "dataset.parquet").exists()


def test_export_artifacts_script_exists_and_is_documented() -> None:
    """The Release escape hatch has to actually be there to be documented.

    `make export-artifacts` is named in `.gitignore` and in docs/DECISIONS.md as
    the path for an artifact too large for git, so the script it shells out to
    must exist and the Makefile target must reference it.
    """
    script = REPO / "scripts" / "export_artifacts.sh"
    assert script.exists(), "scripts/export_artifacts.sh is referenced but absent"
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "export-artifacts:" in makefile
    assert "scripts/export_artifacts.sh" in makefile
    assert "check-artifact-size:" in makefile
