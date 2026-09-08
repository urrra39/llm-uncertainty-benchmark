# configs/

One YAML per run or pilot. Each is a full specification — the model under
test, decoding, prompts, dataset mix and difficulty filters, judge setup,
analysis settings and output paths — so a run is reproducible from its config
alone. Everything loads through the Pydantic `Config` in
`src/unc_bench/config.py` (`extra="forbid"`: a config cannot silently carry a
key the schema does not know). The frozen prompt block is part of the response
cache key, so editing a prompt invalidates cached generations rather than
mixing prompt versions.

Current state, so a stranger does not have to read every file:

- The **published result is run #2b under the fixed label set**:
  `configs/run2b_fixedlabels.yaml` re-analyzes run #2b's rows against the
  corrected labels (`data/run2b/labels_fixed.parquet`) and writes
  `results_run2b_fixedlabels.json`, which the README quotes.
- **Run #3** is pre-registered (`docs/PREREGISTRATION.md`) and pre-flighted in
  `configs/run3_gpu.yaml`, but has never been executed — no number in this
  repository comes from it.

Per-config:

| config | run it describes | status |
|---|---|---|
| `default.yaml` | Run #1's full config (`triviaqa_100`, n=100). | Discarded — run #1's table was sampling noise (tag `run1-n100`); kept as the record and as `run_all.sh`'s `FULL` target. |
| `pilot.yaml` | Run #1's 40-question pilot. | Exploratory, superseded. |
| `run2.yaml` | Run #2 (`run2_easy_mix`, n=120, 90 PopQA / 30 TriviaQA). | Withdrawn — the ranking is withdrawn to `docs/WITHDRAWN_RUN2.md`; its results file is `results_run2_withdrawn.json` and its figures live under `figures/withdrawn_run2/`. |
| `run2_pilot.yaml` | Run #2's first pilot iteration (40 rows). | Exploratory, superseded — base rates recorded in `docs/DECISIONS.md`. |
| `run2_pilot2.yaml` | Run #2's second pilot iteration (40 rows). | Exploratory, superseded — same record. |
| `run2b_clean.yaml` | Run #2b as executed (n=120, 60/60, heuristic labels) — pre-registered in `docs/PREREGISTRATION.md`. | Superseded as the primary by the fixed-label re-analysis below; retained in full as the object of the Round-11 labeler measurement (`results_run2b.json`). |
| `run2b_fixedlabels.yaml` | Run #2b's 120 rows re-labelled under the fixed no-judge rule; NOT a new run. | Current — this is the config that produces the published table (`results_run2b_fixedlabels.json`). Its artifacts are the same rows as `run2b_clean.yaml`, stored once in `data/run2b/`; the config only selects the corrected labels checkpoint (`labels_checkpoint: labels_fixed.parquet`) and the fixed-label human targets. |
| `run3_gpu.yaml` | Run #3 (n=600, 300/300, Qwen2.5-3B-Instruct fp16 on a GPU). | Planned — pre-registered, pre-flighted, never executed. |

`make CONFIG=configs/<name>.yaml <stage>` runs any stage against a config;
see the `Makefile` and the README's reproduction block.
