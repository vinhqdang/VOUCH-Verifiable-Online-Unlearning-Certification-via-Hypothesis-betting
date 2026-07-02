"""Anytime-valid betting machinery for VOUCH.

Implements:
  * One-sided betting e-processes for composite Bernoulli nulls
    (certificate arm  H0: p >= p0,  revocation arm  H0: p <= 1/2),
    with ONS (Online Newton Step), aGRAPA, fixed-lambda, and
    discrete-mixture betting strategies.
  * Waudby-Smith--Ramdas (WSR) betting confidence sequences for a
    bounded mean on a grid over [0, 1].
  * A magnitude-aware symmetry e-process (VOUCH+ revocation arm) that
    bets on sign x magnitude under the exact within-pair symmetry null.

Validity relies only on the supermartingale property under the null and
Ville's inequality; no distributional assumptions on the model or scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "ons_bet",
    "OneSidedEProcess",
    "MixtureEProcess",
    "BettingCS",
    "SymmetryEProcess",
]


def ons_bet(lam: float, z: float, m0: float, a_prev: float,
            eta: float = 0.5, lo: float = 0.0, hi: Optional[float] = None):
    """One Online-Newton-Step update for a wealth factor ``1 + lam * (m0 - z)``.

    Returns the next bet ``lam`` (clipped to ``[lo, hi]``) and the updated
    curvature accumulator ``A``.  Reference implementation from the design
    document (Section 7), following Cutkosky-Orabona ONS for coin betting.
    """
    if hi is None:
        hi = 1.0 / (1.0 - m0) - 1e-6
    g = (m0 - z) / (1.0 + lam * (m0 - z))  # gradient of log-wealth w.r.t. lam
    a_next = a_prev + g * g
    lam_next = min(max(lam + eta * g / a_next, lo), hi)
    return lam_next, a_next


@dataclass
class OneSidedEProcess:
    """Betting e-process for the composite null ``H0: E[Z] >= m0``
    (direction="below", certificate arm: rejecting certifies mean < m0), or
    ``H0: E[Z] <= m0`` (direction="above", revocation arm: rejecting
    certifies mean > m0).  Z must be [0, 1]-valued and the bet predictable.

    Wealth:
        direction "below":  E_t = prod_i (1 + lam_i * (m0 - Z_i)),  lam_i in [0, 1/(1-m0))
        direction "above":  E_t = prod_i (1 + lam_i * (Z_i - m0)),  lam_i in [0, 1/m0)

    Under any P in H0 each factor has conditional mean <= 1, so ``E_t`` is a
    nonnegative supermartingale and Ville's inequality applies at any
    stopping time.
    """

    m0: float
    direction: str = "below"     # "below": evidence that mean < m0
    strategy: str = "mixture"    # "ons" | "agrapa" | "fixed" | "mixture"
    fixed_lam: float = 0.5       # used by strategy="fixed" (as fraction of lam_max)
    eta: float = 0.5
    alpha: float = 0.05
    max_bet_frac: float = 0.999  # stay strictly inside the admissible bet range

    t: int = 0
    log_e: float = 0.0
    _lam: float = 0.0
    _a: float = 1.0
    _sum: float = 0.0
    _sumsq: float = 0.0
    # discrete mixture over fixed bet fractions + ONS expert + KT expert
    _mix_fracs: tuple = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7)
    _mix_logw: np.ndarray = field(default=None)  # type: ignore[assignment]
    log_e_history: list = field(default_factory=list)

    def __post_init__(self):
        if self.direction not in ("below", "above"):
            raise ValueError("direction must be 'below' or 'above'")
        # experts: len(_mix_fracs) fixed-fraction bettors + ONS + KT plug-in
        self._mix_logw = np.zeros(len(self._mix_fracs) + 2)

    # -- geometry -----------------------------------------------------------
    @property
    def lam_max(self) -> float:
        if self.direction == "below":
            return self.max_bet_frac / max(1.0 - self.m0, 1e-12)
        return self.max_bet_frac / max(self.m0, 1e-12)

    def _payoff(self, z: float, lam: float) -> float:
        """Wealth multiplier for one observation with bet lam."""
        if self.direction == "below":
            return 1.0 + lam * (self.m0 - z)
        return 1.0 + lam * (z - self.m0)

    def _kt_payoff(self, z: float) -> float:
        """Truncated Krichevsky-Trofimov plug-in e-factor.

        q = KT estimate of the mean from past data, truncated to the
        alternative side of m0.  Factor  z*q/m0 + (1-z)*(1-q)/(1-m0)  has
        mean <= 1 for every mean on the null side (linear in the mean,
        equal to 1 at m0, decreasing into the null), and for binary z it is
        the Bernoulli likelihood ratio, achieving KL(p, m0) growth up to a
        (1/2) log t redundancy -- near the Theorem-3 optimum.
        """
        q = (self._sum + 0.5) / (self.t + 1.0)
        q = min(q, self.m0) if self.direction == "below" else max(q, self.m0)
        return z * (q / max(self.m0, 1e-12)) + (1.0 - z) * ((1.0 - q) / max(1.0 - self.m0, 1e-12))

    # -- bets ---------------------------------------------------------------
    def _next_lam(self) -> float:
        if self.strategy == "fixed":
            return self.fixed_lam * self.lam_max
        if self.strategy == "ons":
            return self._lam
        if self.strategy == "agrapa":
            if self.t < 2:
                return 0.5 * self.lam_max * 0.1
            mu = self._sum / self.t
            var = max(self._sumsq / self.t - mu * mu, 1e-6)
            signed_gap = (self.m0 - mu) if self.direction == "below" else (mu - self.m0)
            lam = signed_gap / (var + signed_gap * signed_gap)
            return min(max(lam, 0.0), self.lam_max)
        if self.strategy in ("mixture", "kt"):
            return -1.0  # sentinel: handled inside update()
        raise ValueError(f"unknown strategy {self.strategy!r}")

    # -- update -------------------------------------------------------------
    def update(self, z: float) -> float:
        """Observe one Z in [0,1]; returns the current log e-value."""
        z = float(z)
        if self.strategy == "kt":
            step = math.log(max(self._kt_payoff(z), 1e-300))
        elif self.strategy == "mixture":
            # each expert's factor; overall factor = weighted average of
            # expert wealth growth (a mixture of e-processes is an e-process)
            lams = [f * self.lam_max for f in self._mix_fracs] + [self._lam]
            factors = [max(self._payoff(z, l), 1e-300) for l in lams]
            factors.append(max(self._kt_payoff(z), 1e-300))
            factors = np.array(factors)
            w = np.exp(self._mix_logw - self._mix_logw.max())
            w = w / w.sum()
            step = float(np.log(np.dot(w, factors)))
            self._mix_logw += np.log(factors)
            # advance the ONS expert
            self._lam, self._a = ons_bet(
                self._lam, z, self.m0, self._a, eta=self.eta,
                lo=0.0, hi=self.lam_max,
                ) if self.direction == "below" else self._ons_above(z)
        else:
            lam = self._next_lam()
            step = math.log(max(self._payoff(z, lam), 1e-300))
            if self.strategy == "ons":
                if self.direction == "below":
                    self._lam, self._a = ons_bet(self._lam, z, self.m0, self._a,
                                                 eta=self.eta, lo=0.0, hi=self.lam_max)
                else:
                    self._lam, self._a = self._ons_above(z)
        self.t += 1
        self._sum += z
        self._sumsq += z * z
        self.log_e += step
        self.log_e_history.append(self.log_e)
        return self.log_e

    def _ons_above(self, z: float):
        """ONS step for direction='above' (payoff 1 + lam*(z - m0))."""
        g = (z - self.m0) / (1.0 + self._lam * (z - self.m0))
        a = self._a + g * g
        lam = min(max(self._lam + self.eta * g / a, 0.0), self.lam_max)
        return lam, a

    # -- decisions ----------------------------------------------------------
    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e, 700.0))

    def rejects(self, alpha: Optional[float] = None) -> bool:
        a = self.alpha if alpha is None else alpha
        return self.log_e >= math.log(1.0 / a)


class MixtureEProcess:
    """Exponentially-weighted mixture over several e-processes (one per
    score function).  Because a convex combination of e-processes with
    predictable weights is an e-process, validity is exact while power
    adapts to the best score in the class online.

    Used for the *revocation* arm, whose null (Theorem 1: exact unlearning)
    makes every per-score sign exactly Bern(1/2), so every component and any
    predictable mixture is a supermartingale.
    """

    def __init__(self, n_scores: int, m0: float = 0.5, direction: str = "above",
                 strategy: str = "mixture", alpha: float = 0.05, lr: float = 1.0):
        self.components = [
            OneSidedEProcess(m0=m0, direction=direction, strategy=strategy, alpha=alpha)
            for _ in range(n_scores)
        ]
        self.log_mix = 0.0
        self.lr = lr
        self.alpha = alpha
        self.t = 0
        self.log_e_history: list = []

    def update(self, zs) -> float:
        """zs: iterable of per-score Z_i in [0,1] for one pair."""
        zs = list(zs)
        prev = np.array([c.log_e for c in self.components])
        # predictable weights from wealth accumulated so far
        w = np.exp(self.lr * (prev - prev.max()))
        w = w / w.sum()
        for c, z in zip(self.components, zs):
            c.update(z)
        now = np.array([c.log_e for c in self.components])
        factors = np.exp(np.clip(now - prev, -700, 700))
        self.log_mix += float(np.log(np.dot(w, factors)))
        self.t += 1
        self.log_e_history.append(self.log_mix)
        return self.log_mix

    @property
    def log_e(self) -> float:
        return self.log_mix

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_mix, 700.0))

    def rejects(self, alpha: Optional[float] = None) -> bool:
        a = self.alpha if alpha is None else alpha
        return self.log_mix >= math.log(1.0 / a)


class BettingCS:
    """Waudby-Smith--Ramdas betting confidence sequence for a [0,1] mean.

    Maintains, on a grid of candidate means m, two one-sided wealth
    processes (betting up and betting down) with aGRAPA-style predictable
    bets; the running-intersection of ``{m : max(W+, W-) < 2/alpha}`` is a
    (1 - alpha) confidence sequence (union bound over the two sides).
    """

    def __init__(self, alpha: float = 0.05, grid: int = 1001, max_bet_frac: float = 0.75):
        self.alpha = alpha
        self.m = np.linspace(0.0, 1.0, grid)
        self.logw_up = np.zeros(grid)    # bets that true mean > m  (payoff 1 + lam*(z - m))
        self.logw_dn = np.zeros(grid)    # bets that true mean < m  (payoff 1 + lam*(m - z))
        self.t = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.lo = 0.0
        self.hi = 1.0
        self.max_bet_frac = max_bet_frac

    def update(self, z: float) -> None:
        z = float(z)
        if self.t >= 1:
            mu = self._sum / self.t
            var = max(self._sumsq / self.t - mu * mu, 1e-4)
            gap_up = mu - self.m
            gap_dn = self.m - mu
            lam_up = np.clip(gap_up / (var + gap_up ** 2), 0.0,
                             self.max_bet_frac / np.maximum(self.m, 1e-3))
            lam_dn = np.clip(gap_dn / (var + gap_dn ** 2), 0.0,
                             self.max_bet_frac / np.maximum(1.0 - self.m, 1e-3))
        else:
            lam_up = np.zeros_like(self.m)
            lam_dn = np.zeros_like(self.m)
        self.logw_up += np.log(np.maximum(1.0 + lam_up * (z - self.m), 1e-300))
        self.logw_dn += np.log(np.maximum(1.0 + lam_dn * (self.m - z), 1e-300))
        self.t += 1
        self._sum += z
        self._sumsq += z * z
        thresh = math.log(2.0 / self.alpha)
        alive = (self.logw_up < thresh) & (self.logw_dn < thresh)
        if alive.any():
            lo_t, hi_t = float(self.m[alive].min()), float(self.m[alive].max())
        else:  # numerically empty: collapse to the running mean
            mu = self._sum / self.t
            lo_t = hi_t = float(min(max(mu, 0.0), 1.0))
        # running intersection keeps the CS monotone (still a valid CS)
        self.lo = max(self.lo, lo_t)
        self.hi = min(self.hi, hi_t)
        if self.lo > self.hi:
            self.lo = self.hi = 0.5 * (self.lo + self.hi)

    @property
    def interval(self):
        return self.lo, self.hi

    @property
    def advantage_interval(self):
        """CS for the advantage Delta = 2p - 1."""
        return 2.0 * self.lo - 1.0, 2.0 * self.hi - 1.0


class SymmetryEProcess:
    """VOUCH+ magnitude-aware revocation arm.

    Exact null (Theorem 1): under exact unlearning the paired score
    difference D is symmetric about 0, hence  sign(D) | |D|  is a fair coin.
    We bet on the sign with a stake modulated by a predictable, monotone
    transform of |D| (larger |D| = more informative), via wealth factors

        1 + lam * s_i * g_i,   s_i = sign(D_i) in {-1,+1},  g_i in [0,1],

    where g_i = rank of |D_i| among past magnitudes (predictable given
    |D_i| and the past).  E[s_i | |D_i|, past] = 0 under the null, so each
    factor has conditional mean 1: an exact e-process, with power against
    alternatives where memorized pairs produce large positive D.
    """

    def __init__(self, alpha: float = 0.05, eta: float = 0.5, max_bet: float = 0.999):
        self.alpha = alpha
        self.log_e = 0.0
        self.t = 0
        self._lam = 0.0
        self._a = 1.0
        self.eta = eta
        self.max_bet = max_bet
        self._mags: list = []
        self.log_e_history: list = []

    def update(self, d: float, tie_break: float = 0.5) -> float:
        d = float(d)
        if d == 0.0:
            s = 1.0 if tie_break > 0.5 else -1.0
        else:
            s = 1.0 if d > 0 else -1.0
        mag = abs(d)
        if self._mags:
            g = (np.searchsorted(np.sort(self._mags), mag, side="right")) / (len(self._mags) + 1.0)
        else:
            g = 0.5
        x = s * g                      # in [-1, 1], conditional mean 0 under H0
        payoff = 1.0 + self._lam * x
        self.log_e += math.log(max(payoff, 1e-300))
        # ONS on lam in [0, max_bet] (one-sided: alarm on positive advantage)
        grad = x / max(payoff, 1e-12)
        self._a += grad * grad
        self._lam = min(max(self._lam + self.eta * grad / self._a, 0.0), self.max_bet)
        self._mags.append(mag)
        self.t += 1
        self.log_e_history.append(self.log_e)
        return self.log_e

    @property
    def e_value(self) -> float:
        return math.exp(min(self.log_e, 700.0))

    def rejects(self, alpha: Optional[float] = None) -> bool:
        a = self.alpha if alpha is None else alpha
        return self.log_e >= math.log(1.0 / a)
