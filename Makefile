.PHONY: help setup lint fmt type test check pilot all clean-artifacts \
        build_dataset generate score_signals label analyze figures nondeterminism \
        export-artifacts check-artifact-size

UV      ?= uv
RUN     ?= $(UV) run
CONFIG  ?= configs/default.yaml
PILOT   ?= configs/pilot.yaml
# Which run directory the artifact targets operate on. Matches
# `paths.artifacts_dir` in the config you are running.
RUN_DIR ?= data/run3
# Bytes above which a tracked parquet should go to a Release instead of git.
# 50 MB. GitHub's hard per-file limit is 100 MB; this leaves headroom.
SIZE_LIMIT ?= 52428800

help:
	@grep -E '^[a-z_]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

setup: ## create the venv and install pinned deps (add EXTRA=local for torch)
	$(UV) sync --frozen --all-groups
ifeq ($(EXTRA),local)
	$(UV) sync --frozen --all-groups --extra local
endif

lint: ## ruff check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

fmt: ## ruff autofix + format
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

type: ## mypy strict
	$(RUN) mypy

test: ## pytest (fixtures only, no live model calls)
	$(RUN) pytest

check: lint type test ## everything CI runs

# ---- pipeline stages. Each is resumable and idempotent. ----

build_dataset: ## stage 1
	$(RUN) unc-bench build-dataset --config $(CONFIG)

generate: ## stage 2
	$(RUN) unc-bench generate --config $(CONFIG)

score_signals: ## stage 3
	$(RUN) unc-bench score-signals --config $(CONFIG)

label: ## stage 4
	$(RUN) unc-bench label --config $(CONFIG)

analyze: ## stage 5, writes results.json
	$(RUN) unc-bench analyze --config $(CONFIG)

figures: ## regenerate every figure from results.json alone
	$(RUN) unc-bench figures --config $(CONFIG)

nondeterminism: ## rerun 50 greedy prompts twice, report disagreement
	$(RUN) unc-bench nondeterminism --config $(CONFIG)

pilot: ## 100-question sanity pass + the autonomous dataset-mix gate
	$(RUN) unc-bench build-dataset --config $(PILOT)
	$(RUN) unc-bench generate --config $(PILOT)
	$(RUN) unc-bench label --config $(PILOT)
	$(RUN) unc-bench pilot-gate --config $(PILOT)

all: build_dataset generate score_signals label analyze figures ## full run

clean-artifacts: ## drop derived artifacts but KEEP the response cache
	rm -rf data/artifacts figures/*.png results.json

# ---- run artifacts. Tracked in git; see .gitignore and docs/DECISIONS.md. ----

check-artifact-size: ## fail if any tracked artifact in RUN_DIR exceeds SIZE_LIMIT
	@if [ ! -d "$(RUN_DIR)" ]; then \
		echo "$(RUN_DIR) does not exist; nothing to check"; exit 0; \
	fi
	@over=0; \
	for f in $(RUN_DIR)/*.parquet $(RUN_DIR)/*.json; do \
		[ -f "$$f" ] || continue; \
		sz=$$(wc -c <"$$f" | tr -d ' '); \
		printf '  %-40s %10s bytes\n' "$$f" "$$sz"; \
		if [ "$$sz" -gt "$(SIZE_LIMIT)" ]; then \
			echo "    OVER $(SIZE_LIMIT) bytes: commit via 'make export-artifacts' + a Release, not git"; \
			over=1; \
		fi; \
	done; \
	if [ "$$over" -eq 1 ]; then exit 1; fi; \
	echo "all artifacts under $(SIZE_LIMIT) bytes; commit them directly"

export-artifacts: ## package RUN_DIR's artifacts into dist/<run>-artifacts.tar.gz for a Release
	bash scripts/export_artifacts.sh $(RUN_DIR)
