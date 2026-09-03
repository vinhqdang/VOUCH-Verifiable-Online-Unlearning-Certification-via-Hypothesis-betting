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

Two null regimes ("targets")
----------------------------
Both the e-processes and the confidence sequence support two targets, and
the distinction is the subject of Section 4/5 of the paper:

``population_size = None`` -- *super-population* target.  The null is
    ``E[Z_i | F_{i-1}] >= m0`` (resp. ``<= m0``) at every step, i.e. a
    predictable-conditional-mean null.  This is what the revocation arm
    uses: under exact unlearning the pair coin b_i is, conditional on the
    past, still an exact fair coin, so the conditional mean is exactly 1/2
    for every pair regardless of how the signs are correlated across pairs
    (Theorem 1).

``population_size = N`` -- *finite-cohort* (without-replacement) target.
    The null concerns the realised cohort mean ``(1/N) sum_{i<=N} z_i`` of
    the committed cohort of N pairs, and pairs arrive in a committed
    uniformly random order.  The boundary is then re-centred at each step
    on the mean the *unrevealed remainder* would need in order for the
    null to hold,

        m0_t = (N * m0 - sum_{j<t} Z_j) / (N - t + 1),

    and the wealth factor bets against ``m0_t``.  Because sequential
    reveal of a uniform random permutation is simple random sampling
    without replacement, ``E[Z_t | F_{t-1}]`` is exactly the mean of the
    remainder, so the process is a supermartingale under the cohort null
    *with no assumption whatsoever* about independence, exchangeability
    across pairs, or homogeneity across templates and repetition strata.
    This is what the certificate arm uses, and it is the reason the
    certificate survives the dependence induced by all canaries passing
    through one jointly trained and unlearned model.

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

_LOG_INF = 1e6   # stand-in for "the null is refuted with certainty"

# One shared clamp for the recentred boundary.  It MUST be the same everywhere:
# the admissible stake range is lam < 1/(1 - v), so computing lam_max from one
# clamped v and the payoff from another lets the worst-case factor
# 1 + lam*(v - 1) fall below zero.  That happens exactly when the recentred
# boundary reaches 1, which is reachable whenever N*m0 is an integer.
_B_EPS = 1e-9


def ons_bet(lam: float, z: float, m0: float, a_prev: float,
            eta: float = 0.5, lo: float = 0.0, hi: Optional[float] = None):
    """One Online-Newton-Step update for a wealth factor ``1 + lam * (m0 - z)``.

    Returns the next bet ``lam`` (clipped to ``[lo, hi]``) and the updated
    curvature accumulator ``A``, following Cutkosky-Orabona ONS for coin
    betting.
    """
    if hi is None:
        hi = 1.0 / (1.0 - m0) - 1e-6
    g = (m0 - z) / (1.0 + lam * (m0 - z))  # gradient of log-wealth w.r.t. lam
    a_next = a_prev + g * g
    lam_next = min(max(lam + eta * g / a_next, lo), hi)
    return lam_next, a_next


@dataclass
class OneSidedEProcess:
    """Betting e-process for a one-sided Bernoulli-mean null.

    ``direction="below"``  tests ``H0: mean >= m0``  (certificate arm:
        rejecting certifies that the mean is below ``m0``).
    ``direction="above"``  tests ``H0: mean <= m0``  (revocation arm:
        rejecting evidences a mean above ``m0``).

    With ``population_size=None`` "mean" is the predictable conditional
    mean ``E[Z_i | F_{i-1}]`` at every step; with ``population_size=N`` it
    is the realised mean of the committed cohort of N items revealed in a
    uniformly random order (sampling without replacement).  See the module
    docstring.

    Wealth (with ``m0_t`` the effective boundary, equal to ``m0`` in the
    super-population regime):

        "below":  E_t = prod_i (1 + lam_i * (m0_i - Z_i)),  lam_i in [0, 1/(1-m0_i))
        "above":  E_t = prod_i (1 + lam_i * (Z_i - m0_i)),  lam_i in [0, 1/m0_i)

    Under any distribution in the null each factor has conditional mean
    at most 1, so ``E_t`` is a nonnegative supermartingale and Ville's
    inequality applies at every stopping time simultaneously.
    """

    m0: float
    direction: str = "below"     # "below": evidence that the mean < m0
    strategy: str = "mixture"    # "ons" | "agrapa" | "fixed" | "mixture" | "kt"
    fixed_lam: float = 0.5       # used by strategy="fixed" (fraction of lam_max)
    eta: float = 0.5
    alpha: float = 0.05
    max_bet_frac: float = 0.999  # stay strictly inside the admissible bet range
    population_size: Optional[int] = None   # N for the finite-cohort target

    t: int = 0
    log_e: float = 0.0
    _lam: float = 0.0
    _a: float = 1.0
    _sum: float = 0.0
    _sumsq: float = 0.0
    _refuted: bool = False       # cohort null already impossible given the data
    _exhausted: bool = False     # cohort null already implied by the data
    # discrete mixture over fixed bet fractions + ONS expert + KT expert
    _mix_fracs: tuple = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7)
    _mix_logw: np.ndarray = field(default=None)  # type: ignore[assignment]
    log_e_history: list = field(default_factory=list)

    def __post_init__(self):
        if self.direction not in ("below", "above"):
            raise ValueError("direction must be 'below' or 'above'")
        # experts: len(_mix_fracs) fixed-fraction bettors + ONS + KT plug-in
        self._mix_logw = np.zeros(len(self._mix_fracs) + 2)

    # -- effective boundary --------------------------------------------------
    @property
    def m0_t(self) -> float:
        """Boundary the *next* observation is bet against (predictable).

        Super-population regime: the fixed ``m0``.  Finite-cohort regime:
        the mean the unrevealed remainder must have for the cohort null to
        hold, given what has already been revealed.
        """
        n = self.population_size
        if n is None:
            return self.m0
        rem = n - self.t
        if rem <= 0:
            return self.m0
        return (n * self.m0 - self._sum) / rem

    def _boundary_state(self) -> str:
        """'live' | 'refuted' | 'exhausted' for the finite-cohort boundary."""
        if self.population_size is None:
            return "live"
        v = self.m0_t
        if self.direction == "below":
            # H0: remainder mean >= v.  v > 1 is impossible; v <= 0 is free.
            if v > 1.0:
                return "refuted"
            if v <= 0.0:
                return "exhausted"
        else:
            # H0: remainder mean <= v.  v < 0 is impossible; v >= 1 is free.
            if v < 0.0:
                return "refuted"
            if v >= 1.0:
                return "exhausted"
        return "live"

    # -- geometry -----------------------------------------------------------
    def _clamped_boundary(self) -> float:
        """The boundary actually bet against, clamped once into (0, 1)."""
        return min(max(self.m0_t, _B_EPS), 1.0 - _B_EPS)

    def _lam_max_at(self, v: float) -> float:
        """Largest admissible stake at boundary ``v``.

        For direction "below" the wealth factor is 1 + lam*(v - Z) with
        Z in [0,1], whose worst case is Z = 1; requiring it to stay positive
        gives lam < 1/(1 - v).  Mirrored for "above".  ``v`` must be the same
        value the payoff is later evaluated at.
        """
        if self.direction == "below":
            return self.max_bet_frac / (1.0 - v)
        return self.max_bet_frac / v

    @property
    def lam_max(self) -> float:
        return self._lam_max_at(self._clamped_boundary())

    def _payoff(self, z: float, lam: float, v: float) -> float:
        """Wealth multiplier for one observation with bet lam at boundary v."""
        if self.direction == "below":
            return 1.0 + lam * (v - z)
        return 1.0 + lam * (z - v)

    def _kt_payoff(self, z: float, v: float) -> float:
        """Truncated Krichevsky-Trofimov plug-in e-factor at boundary ``v``.

        ``q`` is the KT estimate of the mean from past data, truncated to
        the alternative side of ``v``.  The factor
        ``z*q/v + (1-z)*(1-q)/(1-v)`` has conditional mean <= 1 for every
        mean on the null side (it is linear in the mean, equals 1 at ``v``,
        and decreases into the null); for binary ``z`` it is the Bernoulli
        likelihood ratio, achieving KL(p, v) growth up to a (1/2) log t
        redundancy -- near the Theorem-4 optimum.
        """
        q = (self._sum + 0.5) / (self.t + 1.0)
        q = min(q, v) if self.direction == "below" else max(q, v)
        q = min(max(q, 1e-12), 1.0 - 1e-12)
        vv = min(max(v, 1e-12), 1.0 - 1e-12)
        return z * (q / vv) + (1.0 - z) * ((1.0 - q) / (1.0 - vv))

    # -- bets ---------------------------------------------------------------
    def _next_lam(self, v: float) -> float:
        lam_max = self._lam_max_at(v)
        if self.strategy == "fixed":
            return self.fixed_lam * lam_max
        if self.strategy == "ons":
            return min(self._lam, lam_max)
        if self.strategy == "agrapa":
            if self.t < 2:
                return 0.5 * lam_max * 0.1
            mu = self._sum / self.t
            var = max(self._sumsq / self.t - mu * mu, 1e-6)
            signed_gap = (v - mu) if self.direction == "below" else (mu - v)
            lam = signed_gap / (var + signed_gap * signed_gap)
            return min(max(lam, 0.0), lam_max)
        if self.strategy in ("mixture", "kt"):
            return -1.0  # sentinel: handled inside update()
        raise ValueError(f"unknown strategy {self.strategy!r}")

    # -- update -------------------------------------------------------------
    def update(self, z: float) -> float:
        """Observe one Z in [0,1]; returns the current log e-value."""
        z = float(z)
        state = self._boundary_state()
        if state == "refuted":
            # The finite-cohort null is already incompatible with the data:
            # no more betting is needed, the claim is established.
            self._refuted = True
            self.t += 1
            self._sum += z
            self._sumsq += z * z
            self.log_e = _LOG_INF
            self.log_e_history.append(self.log_e)
            return self.log_e
        if state == "exhausted":
            # The finite-cohort null already holds whatever comes next: any
            # bet would be a bet against a certainty, so stake nothing.
            self._exhausted = True
            self.t += 1
            self._sum += z
            self._sumsq += z * z
            self.log_e_history.append(self.log_e)
            return self.log_e

        v = self._clamped_boundary()
        if self.strategy == "kt":
            step = math.log(max(self._kt_payoff(z, v), 1e-300))
        elif self.strategy == "mixture":
            # each expert's factor; overall factor = weighted average of
            # expert wealth growth (a mixture of e-processes is an e-process)
            lam_max = self._lam_max_at(v)
            lams = [f * lam_max for f in self._mix_fracs] + [min(self._lam, lam_max)]
            factors = [max(self._payoff(z, l, v), 1e-300) for l in lams]
            factors.append(max(self._kt_payoff(z, v), 1e-300))
            factors = np.array(factors)
            w = np.exp(self._mix_logw - self._mix_logw.max())
            w = w / w.sum()
            step = float(np.log(np.dot(w, factors)))
            self._mix_logw += np.log(factors)
            # advance the ONS expert against the current boundary
            self._lam, self._a = (
                ons_bet(min(self._lam, lam_max), z, v, self._a, eta=self.eta,
                        lo=0.0, hi=lam_max)
                if self.direction == "below" else self._ons_above(z, v, lam_max))
        else:
            lam = self._next_lam(v)
            step = math.log(max(self._payoff(z, lam, v), 1e-300))
            if self.strategy == "ons":
                lam_max = self._lam_max_at(v)
                if self.direction == "below":
                    self._lam, self._a = ons_bet(min(self._lam, lam_max), z, v,
                                                 self._a, eta=self.eta,
                                                 lo=0.0, hi=lam_max)
                else:
                    self._lam, self._a = self._ons_above(z, v, lam_max)
        self.t += 1
        self._sum += z
        self._sumsq += z * z
        self.log_e += step
        self.log_e_history.append(self.log_e)
        return self.log_e

    def _ons_above(self, z: float, v: float, lam_max: Optional[float] = None):
        """ONS step for direction='above' (payoff 1 + lam*(z - v))."""
        if lam_max is None:
            lam_max = self.lam_max
        lam0 = min(self._lam, lam_max)
        g = (z - v) / (1.0 + lam0 * (z - v))
        a = self._a + g * g
        lam = min(max(lam0 + self.eta * g / a, 0.0), lam_max)
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
    makes every per-score sign an exact conditional fair coin, so every
    component and any predictable mixture is a supermartingale.
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
    bets; the running intersection of ``{m : max(W+, W-) < 2/alpha}`` is a
    ``(1 - alpha)`` confidence sequence (union bound over the two sides).

    With ``population_size=N`` the target is the realised cohort mean and
    each grid point is bet against its own without-replacement boundary
    ``(N*m - sum_{j<t} Z_j) / (N - t + 1)``; grid points whose boundary
    leaves ``[0, 1]`` are ruled out deterministically, so the interval
    collapses onto the exact cohort mean once the cohort is exhausted.
    """

    def __init__(self, alpha: float = 0.05, grid: int = 1001,
                 max_bet_frac: float = 0.75,
                 population_size: Optional[int] = None):
        self.alpha = alpha
        self.m = np.linspace(0.0, 1.0, grid)
        self.logw_up = np.zeros(grid)    # bets that the true mean > m
        self.logw_dn = np.zeros(grid)    # bets that the true mean < m
        self.t = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.lo = 0.0
        self.hi = 1.0
        self.max_bet_frac = max_bet_frac
        self.population_size = population_size

    def _boundaries(self) -> np.ndarray:
        n = self.population_size
        if n is None:
            return self.m
        rem = n - self.t
        if rem <= 0:
            return self.m
        return (n * self.m - self._sum) / rem

    def update(self, z: float) -> None:
        z = float(z)
        v = self._boundaries()
        feasible = (v >= 0.0) & (v <= 1.0)
        vc = np.clip(v, 1e-9, 1.0 - 1e-9)
        if self.t >= 1:
            mu = self._sum / self.t
            var = max(self._sumsq / self.t - mu * mu, 1e-4)
            gap_up = mu - vc
            gap_dn = vc - mu
            lam_up = np.clip(gap_up / (var + gap_up ** 2), 0.0,
                             self.max_bet_frac / np.maximum(vc, 1e-3))
            lam_dn = np.clip(gap_dn / (var + gap_dn ** 2), 0.0,
                             self.max_bet_frac / np.maximum(1.0 - vc, 1e-3))
        else:
            lam_up = np.zeros_like(self.m)
            lam_dn = np.zeros_like(self.m)
        self.logw_up += np.log(np.maximum(1.0 + lam_up * (z - vc), 1e-300))
        self.logw_dn += np.log(np.maximum(1.0 + lam_dn * (vc - z), 1e-300))
        self.t += 1
        self._sum += z
        self._sumsq += z * z
        thresh = math.log(2.0 / self.alpha)
        alive = (self.logw_up < thresh) & (self.logw_dn < thresh) & feasible
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
    difference D is symmetric about 0 conditionally on the past, hence
    ``sign(D) | |D|`` is a fair coin.  We bet on the sign with a stake
    modulated by a predictable, monotone transform of ``|D|`` (a larger
    magnitude is more informative), via wealth factors

        1 + lam * s_i * g_i,   s_i = sign(D_i) in {-1,+1},  g_i in [0,1],

    where ``g_i`` is the rank of ``|D_i|`` among past magnitudes
    (predictable given ``|D_i|`` and the past).  ``E[s_i | |D_i|, past] = 0``
    under the null, so each factor has conditional mean exactly 1: an exact
    e-process, with power against alternatives in which memorised pairs
    produce large positive D.
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
