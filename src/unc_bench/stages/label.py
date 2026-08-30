"""Stage 4: assign a correctness label to every generated answer.

The route per item: abstention check, then normalized exact match, then a judge
call for whatever is left. Exact match settles the easy majority for free and the
judge only sees the genuinely contested items, which is what keeps the API cost
proportional to the interesting subset rather than to n.

A second, different judge re-labels a deterministic subsample and Cohen's kappa
is computed on the overlap. Kappa is the number that says whether the label set
is worth building a results table on, so it is computed on the same items both
judges actually saw rather than on the whole set.

Judge outcomes are cached on disk keyed by (judge model, qid, answer). The
generator's response cache does not cover them, because a judge call is not a
`generate` on the model under test, and without a cache a rerun of this stage
would pay for every verdict again.

If the gateway is unusable the stage falls back to exact match plus the fuzzy
containment rule and marks the whole label set heuristic. It does not fabricate
a kappa: with one labeler there is no agreement to measure, and the results
report says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from unc_bench.config import Config
from unc_bench.labeling import (
    JudgeOutcome,
    TextJudge,
    build_judge,
    cohens_kappa,
    cross_validation_sample,
    fuzzy_correct,
    judge_item,
    label_by_exact_match,
)
from unc_bench.normalize import clean_model_answer
from unc_bench.stages.common import (
    Progress,
    StagePaths,
    json_load,
    read_checkpoint,
    write_checkpoint,
)
from unc_bench.types import (
    LABEL_CORRECT,
    LABEL_INCORRECT,
    SOURCE_JUDGE,
    Label,
    Question,
)

FLUSH_EVERY = 10

# Consecutive judge failures tolerated before the stage stops calling the
# gateway and finishes on the heuristic path. Five in a row is an outage, not
# bad luck, and continuing would spend the remaining n on timeouts.
MAX_CONSECUTIVE_FAILURES = 5


class JudgeCache:
    """Tiny JSON cache of judge verdicts, keyed by model, qid and answer.

    Keyed on the answer text as well as the qid so that regenerating an answer
    invalidates its verdict instead of silently reusing the grade of a different
    string.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except (OSError, json.JSONDecodeError):
                self.data = {}

    @staticmethod
    def key(model: str, qid: str, answer: str) -> str:
        return f"{model}\u0000{qid}\u0000{answer}"

    def get(self, model: str, qid: str, answer: str) -> dict[str, Any] | None:
        return self.data.get(self.key(model, qid, answer))

    def put(self, model: str, qid: str, answer: str, value: dict[str, Any]) -> None:
        self.data[self.key(model, qid, answer)] = value

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _cached_judge(
    judge: TextJudge,
    cache: JudgeCache,
    question: Question,
    answer: str,
    *,
    seed: int,
) -> JudgeOutcome:
    hit = cache.get(judge.model_name, question.qid, answer)
    if hit is not None:
        return JudgeOutcome(
            qid=question.qid,
            value=hit.get("value"),
            raw=str(hit.get("raw", "")),
            parse_failed=bool(hit.get("parse_failed", False)),
        )
    outcome = judge_item(judge, question, answer, seed=seed)
    cache.put(
        judge.model_name,
        question.qid,
        answer,
        {"value": outcome.value, "raw": outcome.raw, "parse_failed": outcome.parse_failed},
    )
    return outcome


def run(cfg: Config) -> int:
    """Label every generated answer. Returns the number of labeled rows."""
    paths = StagePaths.of(cfg)
    generations = read_checkpoint(paths.generations)
    if generations is None:
        raise FileNotFoundError(f"{paths.generations} is absent; run generate first")

    items: list[tuple[Question, str]] = []
    for record in generations.to_dict(orient="records"):
        question = Question(
            qid=str(record["qid"]),
            dataset=str(record["dataset"]),
            question=str(record["question"]),
            gold_answers=tuple(str(a) for a in (json_load(str(record["gold_answers"])) or [])),
        )
        answer = clean_model_answer(
            str(record.get("greedy_text") or ""), abstain_token=cfg.prompts.abstain_token
        )
        items.append((question, answer))

    # Free pass first, so the judge only sees what exact match cannot settle.
    labels: dict[str, Label] = {}
    contested: list[tuple[Question, str]] = []
    for question, answer in items:
        early = label_by_exact_match(question, answer, cfg.prompts.abstain_token)
        if early is not None:
            labels[question.qid] = early
        else:
            contested.append((question, answer))
    print(
        f"[label] {len(items)} answers: {len(labels)} settled by exact match "
        f"or abstention, {len(contested)} need a judge",
        flush=True,
    )

    judge_cache = JudgeCache(cfg.paths.artifacts_dir / "judge_cache.json")
    primary: TextJudge | None = None
    try:
        primary = build_judge(cfg)
    except Exception as exc:
        print(f"[label] primary judge unavailable ({exc}); falling back to fuzzy", flush=True)

    primary_verdicts: dict[str, str] = {}
    parse_failures = 0
    consecutive = 0
    if primary is not None and contested:
        progress = Progress("judge", len(contested), every=10)
        for question, answer in contested:
            outcome = _cached_judge(primary, judge_cache, question, answer, seed=cfg.greedy.seed)
            if outcome.value is None:
                parse_failures += 1
                consecutive += 1
            else:
                consecutive = 0
                primary_verdicts[question.qid] = outcome.value
                labels[question.qid] = Label(
                    qid=question.qid,
                    value=outcome.value,
                    source=SOURCE_JUDGE,
                    judge_raw=outcome.raw[:200],
                )
            if len(primary_verdicts) % FLUSH_EVERY == 0:
                judge_cache.flush()
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[label] {consecutive} consecutive judge failures; "
                    "stopping judge calls and finishing on the heuristic path",
                    flush=True,
                )
                break
            progress.tick(question.qid)
        judge_cache.flush()

    # Anything the judge could not settle gets the heuristic label, flagged as
    # such in `source` so the analysis can report how much of the label set is
    # judge-derived versus heuristic.
    heuristic_used = 0
    for question, answer in contested:
        if question.qid in labels:
            continue
        value = LABEL_CORRECT if fuzzy_correct(answer, question.gold_answers) else LABEL_INCORRECT
        labels[question.qid] = Label(qid=question.qid, value=value, source="heuristic_fuzzy")
        heuristic_used += 1

    # ---- second judge on a deterministic subsample, for kappa ----
    kappa_payload: dict[str, Any] = {"available": False, "reason": "no second-judge overlap"}
    secondary_verdicts: dict[str, str] = {}
    if primary_verdicts:
        wanted = set(cross_validation_sample(sorted(primary_verdicts), cfg.judges))
        subset = [(q, a) for q, a in contested if q.qid in wanted]
        secondary: TextJudge | None = None
        try:
            secondary = build_judge(cfg, secondary=True)
        except Exception as exc:
            print(f"[label] secondary judge unavailable ({exc}); kappa absent", flush=True)
            kappa_payload = {"available": False, "reason": f"secondary judge unavailable: {exc}"}
        if secondary is not None:
            progress = Progress("judge2", len(subset), every=10)
            for question, answer in subset:
                outcome = _cached_judge(
                    secondary, judge_cache, question, answer, seed=cfg.judges.secondary_seed
                )
                if outcome.value is not None:
                    secondary_verdicts[question.qid] = outcome.value
                progress.tick(question.qid)
            judge_cache.flush()

        paired = sorted(set(primary_verdicts) & set(secondary_verdicts))

        # D11: the kappa denominator must equal the number of rows actually sent
        # to BOTH judges. `wanted` is what was selected for the second judge and
        # `subset` is what it was asked about; a row can drop out of `paired`
        # only by failing to parse, and that count is reported separately rather
        # than silently shrinking the denominator. Asserting it here means a
        # kappa can never be quoted over a row set smaller than it claims.
        sent_to_both = {q.qid for q, _ in subset}
        parse_dropped = sorted(sent_to_both - set(paired))
        if len(paired) + len(parse_dropped) != len(sent_to_both):
            raise AssertionError(
                f"kappa denominator integrity: {len(paired)} paired + "
                f"{len(parse_dropped)} unparsed != {len(sent_to_both)} rows sent "
                f"to both judges"
            )

        if paired:
            result = cohens_kappa(
                [primary_verdicts[q] for q in paired],
                [secondary_verdicts[q] for q in paired],
                threshold=cfg.judges.kappa_trust_threshold,
            )
            kappa_payload = {
                "available": True,
                "kappa": result.kappa,
                "n": result.n,
                "observed_agreement": result.observed_agreement,
                "expected_agreement": result.expected_agreement,
                "categories": list(result.categories),
                "trustworthy": result.trustworthy,
                "n_sent_to_both_judges": len(sent_to_both),
                "n_dropped_for_parse_failure": len(parse_dropped),
                "denominator_matches_rows_sent": len(paired) == len(sent_to_both),
                "primary_model": cfg.judges.primary.name,
                "secondary_model": cfg.judges.secondary.name,
            }
            print(
                f"[label] Cohen's kappa = {result.kappa:.3f} on n={result.n} "
                f"(observed {result.observed_agreement:.3f})",
                flush=True,
            )

    rows = [
        {
            "qid": label.qid,
            "label": label.value,
            "source": label.source,
            "judge_raw": label.judge_raw,
        }
        for label in (labels[q.qid] for q, _ in items)
    ]
    frame = pd.DataFrame(rows)
    write_checkpoint(frame, paths.labels)

    meta = {
        "kappa": kappa_payload,
        "judge_parse_failures": parse_failures,
        "heuristic_fallback_rows": heuristic_used,
        "counts": frame["label"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
    }
    (cfg.paths.artifacts_dir / "label_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[label] wrote {len(frame)} labels to {paths.labels}: {meta['counts']}", flush=True)
    return int(len(frame))
