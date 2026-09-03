"""Score class F for VOUCH verification (Section 4.2).

All scores are computed from temperature-0 (deterministic) forward passes
of the model on the canary text.  The interface is model-agnostic: a model
adapter must expose

    token_logprobs(prefix: str, target: str) -> np.ndarray

returning the per-token log-probabilities of the target span conditioned on
the prefix.  Adapters are provided for the in-repo tiny GPT and for any
HuggingFace causal LM.

Default score class:
    s_loss  : negative token-normalized NLL of the secret span
    s_mink  : min-k% token log-probability (k = 20)
    s_ratio : s_loss(M_u, .) - s_loss(M_0, .)  (base-model calibrated)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "s_loss", "s_mink", "make_s_ratio", "ScoreEngine", "QUERY_WRAPPERS",
    "PARAPHRASE_FRAMES", "s_para",
]


def s_loss(logprobs: np.ndarray) -> float:
    """Negative token-normalized NLL = mean token log-prob of the secret."""
    return float(np.mean(logprobs))


def s_mink(logprobs: np.ndarray, k: float = 0.2) -> float:
    """Min-k% score: mean of the lowest k-fraction token log-probs."""
    n = max(int(np.ceil(k * len(logprobs))), 1)
    return float(np.mean(np.sort(logprobs)[:n]))


def make_s_ratio(base_logprob_fn: Callable[[str, str], np.ndarray]):
    """Returns a score s_ratio(logprobs, prefix, target) that calibrates the
    unlearned-model loss by the base model M_0's loss on the same span."""

    def s_ratio(logprobs: np.ndarray, prefix: str, target: str) -> float:
        base = base_logprob_fn(prefix, target)
        return float(np.mean(logprobs) - np.mean(base))

    return s_ratio


# Q fixed query prompts per twin (Section 6.6): identity + paraphrase
# wrappers around the canary prefix.  Scores per twin are aggregated by the
# mean over wrappers.
QUERY_WRAPPERS: List[Callable[[str], str]] = [
    lambda p: p,
    lambda p: "According to the archives, " + p[0].lower() + p[1:],
    lambda p: "Records state the following. " + p,
    lambda p: "Complete this entry: " + p,
]

# Paraphrase frames for the *paraphrase-aware* score (Section 5.13).
# \citet{li2026beliefs} show that gradient-ascent objectives do not delete
# probability mass so much as move it: mass pushed off one surface form
# reappears on a semantically equivalent rephrasing.  A score that reads the
# literal span under a fixed frame, or averages over frames, cannot see mass
# that has migrated *between* frames -- the average falls even as the fact
# stays fully recoverable under the best frame.  ``s_para`` therefore takes
# the MAXIMUM over frames rather than the mean: it asks whether the secret
# survives under *any* rephrasing of its carrier, which is the quantity an
# attacker who may rephrase actually enjoys.
PARAPHRASE_FRAMES: List[Callable[[str], str]] = [
    lambda p: p,
    lambda p: "Question: " + p.rstrip(".") + "?\nAnswer:",
    lambda p: "It is recorded that " + p[0].lower() + p[1:],
    lambda p: "In other words, " + p[0].lower() + p[1:],
    lambda p: "Restated for the record: " + p,
]


def s_para(frame_logprobs: Sequence[np.ndarray]) -> float:
    """Paraphrase-aware score: best token-normalised log-likelihood of the
    secret over the paraphrase frames.  Higher means more memorised, in the
    same orientation as every other member of F."""
    return float(max(np.mean(lp) for lp in frame_logprobs))


class ScoreEngine:
    """Computes the per-pair score differences D_i^(s) for every score in F.

    Parameters
    ----------
    model_logprob_fn : callable (prefix, target) -> np.ndarray
        Per-token log-probs of ``target`` given ``prefix`` under M_u.
    base_logprob_fn : optional callable, same signature, under M_0.
        Enables the base-calibrated ``ratio`` score.
    n_queries : number of query wrappers Q to aggregate over.
    """

    def __init__(self,
                 model_logprob_fn: Callable[[str, str], np.ndarray],
                 base_logprob_fn: Optional[Callable[[str, str], np.ndarray]] = None,
                 n_queries: int = 4,
                 mink_k: float = 0.2,
                 paraphrase: bool = False,
                 n_frames: int = 5,
                 embed_fn: Optional[Callable[[str, str], float]] = None):
        self.model_fn = model_logprob_fn
        self.base_fn = base_logprob_fn
        self.wrappers = QUERY_WRAPPERS[:max(1, min(n_queries, len(QUERY_WRAPPERS)))]
        self.mink_k = mink_k
        # paraphrase-aware member of F (Section 5.13): off by default so the
        # released tables keep the declared class they were computed under
        self.paraphrase = paraphrase
        self.frames = PARAPHRASE_FRAMES[:max(1, min(n_frames,
                                                    len(PARAPHRASE_FRAMES)))]
        # optional semantic-neighbourhood score: (prefix, secret) -> similarity
        self.embed_fn = embed_fn

    @property
    def score_names(self) -> List[str]:
        names = ["loss", "mink"]
        if self.base_fn is not None:
            names.append("ratio")
        if self.paraphrase:
            names.append("para")
        if self.embed_fn is not None:
            names.append("embed")
        return names

    def _twin_scores(self, prefix: str, target: str) -> Dict[str, float]:
        wrapper_names = [n for n in self.score_names if n not in ("para", "embed")]
        per_wrapper: Dict[str, List[float]] = {n: [] for n in wrapper_names}
        for wrap in self.wrappers:
            wp = wrap(prefix)
            lp = self.model_fn(wp, target)
            per_wrapper["loss"].append(s_loss(lp))
            per_wrapper["mink"].append(s_mink(lp, self.mink_k))
            if self.base_fn is not None:
                base_lp = self.base_fn(wp, target)
                per_wrapper["ratio"].append(float(np.mean(lp) - np.mean(base_lp)))
        out = {n: float(np.mean(v)) for n, v in per_wrapper.items()}
        if self.paraphrase:
            out["para"] = s_para([self.model_fn(f(prefix), target)
                                  for f in self.frames])
        if self.embed_fn is not None:
            out["embed"] = float(self.embed_fn(prefix, target))
        return out

    def pair_differences(self, in_twin, ghost_twin) -> Dict[str, float]:
        """D^(s) = s(M_u, c_in) - s(M_u, c_ghost) for every s in F."""
        s_in = self._twin_scores(*in_twin)
        s_gh = self._twin_scores(*ghost_twin)
        return {n: s_in[n] - s_gh[n] for n in self.score_names}
