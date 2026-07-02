"""Fixed-sample verifier baselines (Section 6.4, item 4) and the peeking
protocol under which their error control breaks.

Baselines operating on the *same* paired-canary signs Z_1..Z_n (this is the
key ablation isolating VOUCH's inference contribution — anytime validity —
from its protocol contribution — the paired-canary design):

  * exact binomial test of H0: p >= p0 (certificate analogue)
  * exact binomial test of H0: p <= 1/2 (revocation analogue)
  * permutation test on paired score differences
  * TOST-style equivalence test via binomial tail
  * two-sample KS test on in vs ghost scores (TOFU "forget quality" analogue)

``peeking_*`` wrappers apply a fixed-n test after every new pair and stop
at the first rejection — the (invalid) way fixed-n tests get used under
streaming deletions and adaptive auditing; Section 6.5 M1 measures the
resulting type-I inflation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

__all__ = [
    "binom_test_cert", "binom_test_rev", "permutation_test", "tost_equivalence",
    "ks_two_sample", "peeking_first_rejection",
]


def binom_test_cert(z: Sequence[float], p0: float) -> float:
    """p-value against H0: p >= p0 (small p-value = evidence unlearned)."""
    z = np.asarray(z)
    return float(stats.binom.cdf(z.sum(), len(z), p0))


def binom_test_rev(z: Sequence[float]) -> float:
    """p-value against H0: p <= 1/2 (small p-value = residual memorization)."""
    z = np.asarray(z)
    return float(stats.binom.sf(z.sum() - 1, len(z), 0.5))


def permutation_test(d: Sequence[float], n_perm: int = 2000, seed: int = 0) -> float:
    """Paired sign-flip permutation test of symmetry (one-sided, mean > 0)."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    return float((1 + np.sum(null >= obs)) / (1 + n_perm))


def tost_equivalence(z: Sequence[float], eps: float) -> float:
    """Equivalence test for the advantage |Delta| < eps via two one-sided
    binomial tests on p in (1/2 - eps/2, 1/2 + eps/2); returns max p-value."""
    z = np.asarray(z)
    n, s = len(z), z.sum()
    p_hi = stats.binom.cdf(s, n, 0.5 + eps / 2)   # H0: p >= 1/2 + eps/2
    p_lo = stats.binom.sf(s - 1, n, 0.5 - eps / 2)  # H0: p <= 1/2 - eps/2
    return float(max(p_hi, p_lo))


def ks_two_sample(scores_in: Sequence[float], scores_ghost: Sequence[float]) -> float:
    """TOFU-style KS test between in-twin and ghost-twin score samples."""
    return float(stats.ks_2samp(scores_in, scores_ghost).pvalue)


def peeking_first_rejection(z: Sequence[float], alpha: float,
                            test: str = "rev", p0: float = 0.55,
                            min_n: int = 5):
    """Apply a fixed-n test after every pair, stop at first rejection.

    Returns (rejected: bool, stop_time: int).  This is the *invalid*
    sequential use of fixed-n tests that VOUCH replaces.
    """
    z = np.asarray(z)
    for t in range(min_n, len(z) + 1):
        if test == "rev":
            p = binom_test_rev(z[:t])
        elif test == "cert":
            p = binom_test_cert(z[:t], p0)
        else:
            raise ValueError(test)
        if p <= alpha:
            return True, t
    return False, len(z)
