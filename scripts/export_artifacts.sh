#!/usr/bin/env bash
#
# Package one run's artifacts into a single compressed archive.
#
# This is the documented ESCAPE HATCH, not the primary path. Run artifacts are
# tracked in git directly (see .gitignore and docs/DECISIONS.md, "Run artifacts
# are tracked from run #3 on"), because the measured worst case for a 600-row
# run is 6.68 MB — two orders of magnitude under GitHub's 100 MB hard limit and
# well under the ~50 MB threshold at which Git LFS becomes worth its cost.
#
# Use this target when that stops being true: a run with a much larger n, a
# longer max_new_tokens, or a wider top_logprobs could push generations.parquet
# past the threshold. The archive is intended for attachment to a GitHub
# Release, so the numbers stay retrievable without putting a large binary in the
# repository's history, which is not removable afterwards without a rewrite.
#
# Usage:
#   scripts/export_artifacts.sh data/run3            # -> dist/run3-artifacts.tar.gz
#   scripts/export_artifacts.sh data/run3 dist/out   # explicit output directory
#
# Deliberately does NOT include: the response cache, the judge cache, the
# downloaded source corpora, or model weights. Those are regenerable and the
# first three are large.

set -euo pipefail

RUN_DIR="${1:-}"
OUT_DIR="${2:-dist}"

if [ -z "$RUN_DIR" ]; then
    echo "usage: $0 <run-artifacts-dir> [output-dir]" >&2
    echo "example: $0 data/run3" >&2
    exit 2
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "error: $RUN_DIR is not a directory" >&2
    exit 1
fi

RUN_NAME="$(basename "$RUN_DIR")"
ARCHIVE="$OUT_DIR/${RUN_NAME}-artifacts.tar.gz"

# The tracked artifact set, in the order docs/DECISIONS.md lists it. A missing
# member is reported and skipped rather than aborting: a run that stopped after
# generation has no labels.parquet, and its generations are still worth keeping.
MEMBERS=(
    dataset.parquet
    generations.parquet
    signals_actc.parquet
    signals_b.parquet
    labels.parquet
    judge_verdicts.json
    timings.json
    pilot_gate.json
)

PRESENT=()
for member in "${MEMBERS[@]}"; do
    if [ -f "$RUN_DIR/$member" ]; then
        PRESENT+=("$member")
    else
        echo "  absent, skipped: $member"
    fi
done

if [ "${#PRESENT[@]}" -eq 0 ]; then
    echo "error: $RUN_DIR holds none of the expected artifacts" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
tar -czf "$ARCHIVE" -C "$RUN_DIR" "${PRESENT[@]}"

BYTES="$(wc -c <"$ARCHIVE" | tr -d ' ')"
echo "wrote $ARCHIVE (${BYTES} bytes, ${#PRESENT[@]} members)"
echo
echo "To publish, attach it to a GitHub Release:"
echo "  gh release create ${RUN_NAME} $ARCHIVE --notes 'run artifacts for ${RUN_NAME}'"
echo
echo "Then record the release URL in docs/DECISIONS.md so a reader can find it."
