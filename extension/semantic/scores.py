"""Scores for semantic twins.

All scores compare the two values of a pair under the *same* prompt, so each is
a function of the model and the pair only and stays inside the declared class F.

  literal      mean token log-prob of the value after the literal training prefix
  belief       the same, averaged over the paraphrased questions (never trained)
  belief_max   the maximum over paraphrased questions (VOUCH's s_para analogue)
  *_cal        the same score minus the base model's score, which removes the
               base model's prior preference between the two values
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .canaries import SemanticPair

LogprobFn = Callable[[str, str], Sequence[float]]

__all__ = ["pair_score_diffs", "SCORE_NAMES"]

SCORE_NAMES = ("literal", "belief", "belief_max")


def _value_scores(fn: LogprobFn, pair: SemanticPair, value: str,
                  questions: Optional[Sequence[int]] = None) -> Dict[str, float]:
    target = " " + value
    lit = float(np.mean(fn(pair.literal_prefix(), target)))
    qp = pair.question_prefixes()
    idx = range(len(qp)) if questions is None else questions
    qs = [float(np.mean(fn(qp[k], target))) for k in idx]
    return {"literal": lit, "belief": float(np.mean(qs)), "belief_max": float(np.max(qs))}


def pair_score_diffs(fn: LogprobFn, pair: SemanticPair,
                     base_fn: Optional[LogprobFn] = None,
                     questions: Optional[Sequence[int]] = None,
                     base_cache: Optional[dict] = None) -> Dict[str, float]:
    """Score difference in-twin minus ghost-twin for every score.

    ``questions`` restricts the belief scores to question indices never used
    in training. ``base_cache`` memoises base-model scores across subjects."""
    s_in = _value_scores(fn, pair, pair.in_value, questions)
    s_gh = _value_scores(fn, pair, pair.ghost_value, questions)
    out = {k: s_in[k] - s_gh[k] for k in SCORE_NAMES}
    if base_fn is not None:
        key = (pair.pair_id, pair.entity)
        if base_cache is not None and key in base_cache:
            b_in, b_gh = base_cache[key]
        else:
            b_in = _value_scores(base_fn, pair, pair.in_value, questions)
            b_gh = _value_scores(base_fn, pair, pair.ghost_value, questions)
            if base_cache is not None:
                base_cache[key] = (b_in, b_gh)
        for k in SCORE_NAMES:
            out[k + "_cal"] = out[k] - (b_in[k] - b_gh[k])
    return out
