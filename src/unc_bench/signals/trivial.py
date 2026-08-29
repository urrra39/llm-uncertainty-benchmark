"""Family T: the baselines that decide whether anything else mattered.

Cost is zero. These are not a footnote to the results table; they are the control
the rest of the table is measured against.

Answer length is the one that matters. Mean logprob and answer length are
mechanically coupled: each additional token contributes another negative logprob,
so a short answer has a higher mean almost by construction. If `a_mean_logprob`
reaches AUROC 0.68 and `t_answer_length` reaches 0.66, the honest reading is that
family A mostly rediscovered answer length at 5x the code. That comparison is
only available if the baseline is in the table.

Random is the null. Its CI should contain 0.50, and if it does not, the
bootstrap is wrong rather than the coin being lucky.
"""

from __future__ import annotations

import hashlib

import numpy as np

from unc_bench.normalize import tokenize
from unc_bench.signals.base import (
    FAMILY_T,
    ORIENT_RISK,
    SignalSpec,
    register,
)

ANSWER_LENGTH = register(
    SignalSpec(
        name="t_answer_length",
        family=FAMILY_T,
        orientation=ORIENT_RISK,
        description="normalized answer token count",
    )
)
QUESTION_LENGTH = register(
    SignalSpec(
        name="t_question_length",
        family=FAMILY_T,
        orientation=ORIENT_RISK,
        description="normalized question token count",
    )
)
RANDOM = register(
    SignalSpec(
        name="t_random",
        family=FAMILY_T,
        orientation=ORIENT_RISK,
        description="seeded uniform noise; the null signal",
    )
)

FAMILY_T_SIGNALS: tuple[SignalSpec, ...] = (ANSWER_LENGTH, QUESTION_LENGTH, RANDOM)


def compute_family_t(question: str, answer: str, *, qid: str, seed: int) -> dict[str, float]:
    """Every family-T signal, in raw units.

    The random value is derived from a hash of `(seed, qid)` rather than drawn
    from a shared stream. A shared stream would make every row's noise depend on
    the order the rows were processed in, so adding one question upstream would
    change the random baseline for every question after it and the number would
    not be reproducible from the config alone.

    The hash is SHA-256, not the builtin `hash()`. Python salts string hashing
    per process unless PYTHONHASHSEED is set, so `hash(qid)` returns a different
    value on every invocation and the "reproducible" baseline would silently be
    a fresh draw each run. This is the sort of thing that only shows up when two
    runs of the same config disagree on one column.

    Both lengths are counted in NORMALIZED tokens. Counting raw characters would
    make the baseline partly a measure of punctuation and casing, which is not
    the confound being controlled for.

    Orientation for the lengths is `risk`: longer answers are hypothesized to be
    likelier wrong, since a small model padding out an answer it does not have is
    the common failure. If the real effect runs the other way the AUROC lands
    below 0.5 and says so, which is a result rather than a bug.
    """
    digest = hashlib.sha256(f"{seed}:{qid}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return {
        ANSWER_LENGTH.name: float(len(tokenize(answer))),
        QUESTION_LENGTH.name: float(len(tokenize(question))),
        RANDOM.name: float(rng.random()),
    }
