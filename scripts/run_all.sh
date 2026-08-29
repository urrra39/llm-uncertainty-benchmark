#!/usr/bin/env bash
# Drives the pilot to completion, records the gate, then runs the full study.
#
# One script rather than a Makefile chain because the two runs have to be
# sequenced with the generator unloaded between the generate and family-B
# stages, and because a single log file is easier to reason about after the
# fact than four interleaved ones.
#
# Every stage is resumable, so re-running this script after an interrupt costs
# only the questions that had not finished.
set -uo pipefail
cd "$(dirname "$0")/.."

PILOT=configs/pilot.yaml
FULL=configs/default.yaml
LOG=logs
mkdir -p "$LOG"

step() { echo "=== $(date -u +%H:%M:%S) $* ===" ; }

# ---- pilot: finish generation, score, label, gate ----
step "pilot generate"
uv run unc-bench generate --config "$PILOT"          >> "$LOG/pilot_generate.log" 2>&1
step "pilot score-signals actc"
uv run unc-bench score-signals --config "$PILOT" --family actc >> "$LOG/pilot_signals.log" 2>&1
step "pilot score-signals b"
uv run unc-bench score-signals --config "$PILOT" --family b    >> "$LOG/pilot_signals.log" 2>&1
step "pilot label"
uv run unc-bench label --config "$PILOT"             >> "$LOG/pilot_label.log" 2>&1
step "pilot gate"
uv run unc-bench pilot-gate --config "$PILOT"        >> "$LOG/pilot_gate.log" 2>&1
cat data/pilot/pilot_gate.json

# ---- full run ----
step "full build-dataset"
uv run unc-bench build-dataset --config "$FULL"      >> "$LOG/full_build.log" 2>&1
step "full generate"
uv run unc-bench generate --config "$FULL"           >> "$LOG/full_generate.log" 2>&1
step "full score-signals actc"
uv run unc-bench score-signals --config "$FULL" --family actc >> "$LOG/full_signals.log" 2>&1
step "full score-signals b"
uv run unc-bench score-signals --config "$FULL" --family b    >> "$LOG/full_signals.log" 2>&1
step "full label"
uv run unc-bench label --config "$FULL"              >> "$LOG/full_label.log" 2>&1
step "full analyze"
uv run unc-bench analyze --config "$FULL"            >> "$LOG/full_analyze.log" 2>&1
step "done"
ls -la figures/ results.json
