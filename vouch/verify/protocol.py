"""VOUCH Phase-2 certification protocol (Section 4.7 pseudocode).

Runs, over canary pairs revealed in random order:

  (a) the certificate e-process   — sequential equivalence test of
      H0^cert : Delta >= eps   (issue certificate when rejected);
  (b) the advantage confidence sequence [L_t, U_t] (WSR betting CS);
  (c) the revocation e-process    — alarm on H0^rev : Delta <= 0.

Soundness refinement over the v1.0 design document
--------------------------------------------------
The design doc bets the *certificate* arm on the mixture-weighted sign
Zbar_i = sum_s w_s Z_i^s.  That is valid for the revocation arm (whose null
makes every Z_i^s exactly Bern(1/2), hence any predictable mixture has
conditional mean 1/2), but NOT in general for the certificate arm: its
composite null is  "exists s in F with p^s >= p0",  under which the mixture
mean can be below p0, breaking the supermartingale property.

VOUCH therefore runs one certificate e-process per score s in F and issues
the certificate only when  min_s E_t^{cert,s} >= 1/alpha.  A false
certificate requires the e-process of a truly-violating score to cross
1/alpha, which has probability <= alpha by Ville — no Bonferroni needed.
The same argument makes  max_s U_t^s  a valid anytime upper bound on
sup_s p^s.  Power adaptivity is preserved: each per-score process bets with
ONS/mixture on its own stream, and the revocation arm keeps the
exponentially-weighted mixture (plus a magnitude-aware symmetry e-process,
"VOUCH+", which exploits |D| under the exact symmetry null).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .betting import (BettingCS, MixtureEProcess, OneSidedEProcess,
                      SymmetryEProcess)

__all__ = ["VouchConfig", "VouchVerifier", "Certificate", "GlobalCertificate"]


@dataclass
class VouchConfig:
    eps: float = 0.10            # certified bound on the residual advantage
    alpha: float = 0.05          # error level
    strategy: str = "mixture"    # betting strategy for e-processes
    use_magnitude_revocation: bool = True   # VOUCH+ arm
    two_sided: bool = True       # close F under negation (catches over-forgetting)
    tie_seed: int = 20260702     # committed PRNG seed for tie-breaking
    cs_grid: int = 1001

    @property
    def p0(self) -> float:
        """Sign-probability boundary: Delta = eps  <=>  p = 1/2 + eps/2."""
        return 0.5 + self.eps / 2.0


@dataclass
class Certificate:
    """The published certificate object (Section 7)."""
    status: str                  # "ISSUED" | "REVOKED" | "UNDETERMINED"
    eps: float
    alpha: float
    wave: int
    t_stop: int
    t_revoked: int               # first revocation-crossing time (-1 if none)
    log_e_cert: float            # min over scores at t_stop
    log_e_rev: float             # final value
    log_e_rev_max: float         # running max (value at the revocation decision)
    p_upper: float               # anytime upper conf. bound on sup_s p^s
    delta_upper: float           # = 2 * p_upper - 1
    delta_cs: Dict[str, List[float]]
    per_score_log_e_cert: Dict[str, float]
    manifest_sha256: str = ""
    score_class: List[str] = field(default_factory=list)
    probes: Dict[str, dict] = field(default_factory=dict)
    code_version: str = "vouch-1.0"

    def to_json(self) -> str:
        d = dict(self.__dict__)
        return json.dumps(d, indent=2, sort_keys=True, default=float)


class VouchVerifier:
    """Runs the Phase-2 loop over one canary cohort (one deletion wave)."""

    def __init__(self, score_names: Sequence[str], config: Optional[VouchConfig] = None,
                 wave: int = 0, manifest_sha256: str = ""):
        self.cfg = config or VouchConfig()
        self.score_names = list(score_names)
        self.wave = wave
        self.manifest_sha256 = manifest_sha256
        p0, a = self.cfg.p0, self.cfg.alpha
        # (a) per-score certificate e-processes: H0^cert,s : p^s >= p0.
        # With two_sided=True, F is closed under negation: each score also
        # gets a process against H0 : p^s <= 1 - p0, so the certificate
        # asserts |Delta^s| < eps for every s (a below-chance in-twin score
        # is membership leakage too -- "over-forgetting", observed for
        # ascent-style unlearning on real LMs).
        self.e_cert = {s: OneSidedEProcess(m0=p0, direction="below",
                                           strategy=self.cfg.strategy, alpha=a)
                       for s in self.score_names}
        self.e_cert_neg = {}
        if self.cfg.two_sided:
            self.e_cert_neg = {s: OneSidedEProcess(m0=1.0 - p0, direction="above",
                                                   strategy=self.cfg.strategy, alpha=a)
                               for s in self.score_names}
        # (c) revocation arm: sign-mixture across scores, H0^rev : p^s <= 1/2
        self.e_rev_sign = MixtureEProcess(n_scores=len(self.score_names),
                                          m0=0.5, direction="above",
                                          strategy=self.cfg.strategy, alpha=a)
        self.e_rev_sign_dn = MixtureEProcess(n_scores=len(self.score_names),
                                             m0=0.5, direction="below",
                                             strategy=self.cfg.strategy, alpha=a) \
            if self.cfg.two_sided else None
        # (c') VOUCH+ magnitude-aware revocation (per score, mixed uniformly)
        self.e_rev_mag = {s: SymmetryEProcess(alpha=a) for s in self.score_names}
        self.e_rev_mag_dn = {s: SymmetryEProcess(alpha=a) for s in self.score_names} \
            if self.cfg.two_sided else None
        # (b) per-score confidence sequences for p^s
        self.cs = {s: BettingCS(alpha=a, grid=self.cfg.cs_grid)
                   for s in self.score_names}
        self._tie_rng = random.Random(self.cfg.tie_seed)
        self.t = 0
        self.revoked_at: Optional[int] = None
        self.log_e_rev_max: float = 0.0   # running max (value at the decision)
        self.history: List[dict] = []

    # -- revocation combination ---------------------------------------------
    @property
    def log_e_rev(self) -> float:
        """Average of the sign-mixture and magnitude e-processes (an average
        of e-processes is an e-process)."""
        les = [self.e_rev_sign.log_e]
        if self.cfg.two_sided:
            les.append(self.e_rev_sign_dn.log_e)
        if self.cfg.use_magnitude_revocation:
            les.extend(m.log_e for m in self.e_rev_mag.values())
            if self.cfg.two_sided:
                les.extend(m.log_e for m in self.e_rev_mag_dn.values())
        mx = max(les)
        return mx + math.log(sum(math.exp(le - mx) for le in les) / len(les))

    @property
    def log_e_cert(self) -> float:
        vals = [e.log_e for e in self.e_cert.values()]
        vals += [e.log_e for e in self.e_cert_neg.values()]
        return min(vals)

    @property
    def p_upper(self) -> float:
        return max(c.hi for c in self.cs.values())

    # -- one pair ------------------------------------------------------------
    def update(self, diffs: Dict[str, float]) -> dict:
        """Observe the score differences D_i^(s) for one pair."""
        zs = {}
        for s in self.score_names:
            d = diffs[s]
            tie = self._tie_rng.random()
            z = float(d > 0) if d != 0 else float(tie > 0.5)
            zs[s] = z
            self.e_cert[s].update(z)
            if self.cfg.two_sided:
                self.e_cert_neg[s].update(z)
                self.e_rev_mag_dn[s].update(-d, tie_break=1.0 - tie)
            self.cs[s].update(z)
            self.e_rev_mag[s].update(d, tie_break=tie)
        self.e_rev_sign.update([zs[s] for s in self.score_names])
        if self.cfg.two_sided:
            self.e_rev_sign_dn.update([zs[s] for s in self.score_names])
        self.t += 1
        self.log_e_rev_max = max(self.log_e_rev_max, self.log_e_rev)
        state = {
            "t": self.t,
            "log_e_cert": self.log_e_cert,
            "log_e_rev": self.log_e_rev,
            "p_upper": self.p_upper,
        }
        self.history.append(state)
        return state

    # -- decisions -----------------------------------------------------------
    @property
    def threshold(self) -> float:
        return math.log(1.0 / self.cfg.alpha)

    def certificate_earned(self) -> bool:
        return self.log_e_cert >= self.threshold

    def revoked(self) -> bool:
        return self.log_e_rev >= self.threshold

    # -- full loop -----------------------------------------------------------
    def run(self, pair_diffs: Sequence[Dict[str, float]],
            shuffle_seed: Optional[int] = None,
            early_stop: bool = True) -> Certificate:
        """Run the Phase-2 loop over a cohort of pair score-differences.

        ``pair_diffs`` is a list of {score_name: D_i^(s)} dicts.  Pairs are
        revealed in random order (predictable filtration).
        """
        order = list(range(len(pair_diffs)))
        rng = random.Random(self.cfg.tie_seed if shuffle_seed is None else shuffle_seed)
        rng.shuffle(order)
        status = "UNDETERMINED"
        for idx in order:
            self.update(pair_diffs[idx])
            if status != "REVOKED" and self.revoked():
                status = "REVOKED"
                self.revoked_at = self.t
                # do NOT break: keep processing the cohort so the revocation
                # e-process compounds evidence for the streaming global alarm
                # (scoring is already paid for; stopping is optional anyway).
            if status == "UNDETERMINED" and self.certificate_earned() and early_stop:
                status = "ISSUED"
                break
        if status == "UNDETERMINED":
            if self.revoked():
                status = "REVOKED"
            elif self.certificate_earned():
                status = "ISSUED"
        return self._certificate(status)

    def _certificate(self, status: str) -> Certificate:
        return Certificate(
            status=status,
            eps=self.cfg.eps,
            alpha=self.cfg.alpha,
            wave=self.wave,
            t_stop=self.t,
            t_revoked=self.revoked_at if self.revoked_at is not None else -1,
            log_e_cert=self.log_e_cert,
            log_e_rev=self.log_e_rev,
            log_e_rev_max=self.log_e_rev_max,
            p_upper=self.p_upper,
            delta_upper=2.0 * self.p_upper - 1.0,
            delta_cs={s: list(self.cs[s].advantage_interval) for s in self.score_names},
            per_score_log_e_cert={s: self.e_cert[s].log_e for s in self.score_names},
            manifest_sha256=self.manifest_sha256,
            score_class=self.score_names,
        )


class GlobalCertificate:
    """Streaming composition across deletion waves (Section 4.5), with the
    composition direction corrected relative to the v1.0 design document.

    Certificates and alarms compose *differently* because their nulls sit on
    opposite sides:

      * Global certificate ("every wave sufficiently unlearned") targets a
        UNION null (some wave has Delta >= eps).  The sound composition is
        ALL-PASS: every wave individually earns its certificate at level
        alpha.  A false global certificate then requires a truly-bad wave's
        own e-process to cross 1/alpha -- probability <= alpha by Ville,
        regardless of how many waves the history contains.  (Multiplying
        certificate e-values, as Theorem 5 of the design doc suggests,
        tests the INTERSECTION null "every wave is bad" -- rejecting it
        only certifies that *some* wave is clean, which is not the claim.)

      * Global revocation alarm ("some residual memorization somewhere")
        targets the INTERSECTION null (all waves exactly unlearned), under
        which every per-wave revocation e-process is a supermartingale, so
        their product composes soundly: alarm when the product exceeds
        1/alpha.  This history-wide alarm accumulates distributed,
        sub-threshold leakage that no single wave reveals, and its
        false-alarm probability over the entire deletion history is
        <= alpha at any time.

    Per-wave alarms additionally use alpha-spending alpha_k = alpha * 2^-(k+1)
    so that family-wise false revocation over an unbounded history stays
    <= alpha while each wave retains an individual alarm.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.log_e_rev_global = 0.0
        self.waves: List[Certificate] = []

    def add_wave(self, cert: Certificate) -> None:
        self.waves.append(cert)
        self.log_e_rev_global += cert.log_e_rev

    @property
    def alpha_k(self) -> float:
        """Alpha-spending level for the next wave's individual alarm."""
        return self.alpha * 2.0 ** (-(len(self.waves) + 1))

    @property
    def certified(self) -> bool:
        """All-pass composition: every wave issued its certificate."""
        return bool(self.waves) and all(c.status == "ISSUED" for c in self.waves)

    @property
    def revocation_alarm(self) -> bool:
        """History-wide anytime-valid alarm at level alpha (product)."""
        return self.log_e_rev_global >= math.log(1.0 / self.alpha)

    def summary(self) -> dict:
        return {
            "waves": len(self.waves),
            "log_e_rev_global": self.log_e_rev_global,
            "certified": self.certified,
            "revocation_alarm": self.revocation_alarm,
            "statuses": [c.status for c in self.waves],
            "manifest_hashes": [c.manifest_sha256 for c in self.waves],
        }


def commit(data: str) -> str:
    """SHA-256 commitment helper."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
