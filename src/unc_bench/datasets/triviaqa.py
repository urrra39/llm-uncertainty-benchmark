"""TriviaQA, `rc.nocontext` validation split.

Trivia-style questions written by humans, with a long alias list per answer. The
`nocontext` configuration strips the retrieved evidence documents, which is what
this benchmark wants: the question is whether the model knows the fact, not
whether it can read.

17,944 rows in the validation split. Read from the single parquet file the Hub
serves for this configuration rather than through the `datasets` library, which
would add a dependency and a second cache layer for one flat file. Same reasoning
as PopQA (docs/DECISIONS.md D10).

`answer.aliases` and `answer.normalized_aliases` are both kept. The normalized
list is lowercased and punctuation-stripped by the dataset authors, which is
close to but not identical to this project's `normalize_answer`, so both are
passed through and the project's own normalizer runs over all of them at match
time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from unc_bench.datasets.base import DatasetBuilder, cached_download, clean_alias_list
from unc_bench.types import Question

# The Hub's parquet endpoint for this configuration returns exactly one shard.
TRIVIAQA_URL = (
    "https://huggingface.co/api/datasets/mandarjoshi/trivia_qa/"
    "parquet/rc.nocontext/validation/0.parquet"
)


class TriviaQABuilder(DatasetBuilder):
    name = "triviaqa"

    #: When True, keep only questions whose answer entity looks well-known.
    #: See `_looks_high_frequency` for what that means and why it is a proxy.
    easy_only: bool = False

    def __init__(self, raw_dir: Path, *, easy_only: bool = False) -> None:
        super().__init__(raw_dir)
        self.easy_only = easy_only

    @property
    def local_path(self) -> Path:
        return self.raw_dir / "triviaqa_rc_nocontext_validation.parquet"

    def load_candidates(self) -> list[Question]:
        path = cached_download(TRIVIAQA_URL, self.local_path)
        frame = pd.read_parquet(path, columns=["question", "question_id", "answer"])

        required = {"question", "question_id", "answer"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"triviaqa: source is missing columns {sorted(missing)}")

        questions = frame["question"].tolist()
        qids = frame["question_id"].tolist()
        answers = frame["answer"].tolist()

        out: list[Question] = []
        # The `rc.nocontext` validation split repeats a `question_id` whenever the
        # same question was paired with more than one evidence document upstream.
        # Stripping the context collapses those rows into identical questions, so
        # the first occurrence of each id is kept and the rest dropped. Without
        # this, `questions_to_frame` rejects the whole draw on duplicate qids.
        seen: set[str] = set()
        for question, qid, answer in zip(questions, qids, answers, strict=True):
            text = str(question).strip()
            if not text:
                continue
            aliases = _extract_aliases(answer)
            if not aliases:
                continue
            key = f"triviaqa-{qid}"
            if key in seen:
                continue
            if self.easy_only and not _looks_high_frequency(text, aliases):
                continue
            seen.add(key)
            out.append(
                Question(
                    qid=f"triviaqa-{qid}",
                    dataset=self.name,
                    question=text,
                    gold_answers=aliases,
                )
            )
        return out


#: Alias-count threshold for "well-known". TriviaQA's alias list is built from
#: Wikipedia redirects and surface forms, so a heavily-redirected entity has many
#: aliases and an obscure one has few. This is a proxy for entity frequency and
#: it is a proxy: TriviaQA ships no popularity field, unlike PopQA's `s_pop`.
#: The substitution is documented in docs/DECISIONS.md.
MIN_ALIASES_FOR_EASY = 12

#: Upper bound on question length in words. Long trivia questions are long
#: because they pile on qualifying clauses, and a 0.5B model loses the thread.
#: Also part of the frequency proxy rather than an independent criterion.
MAX_QUESTION_WORDS_FOR_EASY = 20


def _looks_high_frequency(question: str, aliases: tuple[str, ...]) -> bool:
    """Proxy test for "this asks about a well-known entity".

    Three conditions, all of which have to hold:

    - the answer has at least `MIN_ALIASES_FOR_EASY` aliases, i.e. Wikipedia
      redirects a lot of surface forms onto it
    - the question is at most `MAX_QUESTION_WORDS_FOR_EASY` words
    - the primary alias is at most four words, which excludes the long
      descriptive answers ("the Treaty of Brest-Litovsk of March 1918") that a
      24-token greedy decode cannot produce even when the model knows the fact

    This raises the base rate, which is the whole point (docs/DECISIONS.md, run
    #2 D1). It also biases the question set, and that bias is stated in
    LIMITATIONS.md rather than hidden: the resulting benchmark measures error
    prediction on popular-entity trivia, not on trivia in general.
    """
    if len(aliases) < MIN_ALIASES_FOR_EASY:
        return False
    if len(question.split()) > MAX_QUESTION_WORDS_FOR_EASY:
        return False
    return len(aliases[0].split()) <= 4


def _extract_aliases(answer: Any) -> tuple[str, ...]:
    """Pull the canonical value and every alias out of one answer struct.

    `value` goes first so the primary alias is the surface form a human would
    write, matching the PopQA builder's convention. Order matters only for
    display; matching tries every alias.

    The alias arrays arrive as numpy arrays inside the parquet struct, not lists,
    so they are coerced explicitly. A bare `if not aliases` on a numpy array
    raises "truth value of an array is ambiguous", which is a confusing failure
    to debug from inside a dataset builder.
    """
    if not isinstance(answer, dict):
        return ()
    parts: list[str] = []
    value = answer.get("value")
    if value is not None:
        parts.append(str(value))
    for key in ("aliases", "normalized_aliases"):
        raw = answer.get(key)
        if raw is None:
            continue
        parts.extend(str(a) for a in list(raw))
    return clean_alias_list(parts)
