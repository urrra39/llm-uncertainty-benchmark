# Label gate map

Generated from `GATE_SOURCES` plus `configs/run2b_fixedlabels.yaml` — edit the code, not this file. Pinned by test.

| gate | reads | denominator | threshold | labelling it unlocks |
|---|---|---|---|---|
| labeling_protocol_validated | data/human_validation_sample_run2b_fixed.csv (the run's own validation sample (cfg.paths.human_validation_csv)) | validation sample rows | MIN_PROTOCOL_COVERAGE = 0.50 | instructions exercised; further labelling proceeds on tested wording |
| human_label_coverage | data/fuzzy_decided_rows_fixed.csv (the run's fuzzy-decided rows (cfg.paths.fuzzy_decided_csv)) | fuzzy-decided rows | MIN_HUMAN_LABEL_COVERAGE = 0.80 | publishable ranking (with all other gates) |

Every file above belongs to the run being gated. Labelling withdrawn rows earns no gate credit by construction.
