"""Unit tests for the anytime-valid machinery.

Run:  python3 -m pytest tests/ -q   (or python3 tests/test_betting.py)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.canaries import PGCGenerator, CanaryManifest
from vouch.verify import (BettingCS, MixtureEProcess, OneSidedEProcess,
                          SymmetryEProcess, VouchConfig, VouchVerifier)


def test_eprocess_supermartingale_null():
    """Under the null boundary p = p0, E[E_t] <= 1 (Monte Carlo)."""
    rng = np.random.default_rng(0)
    p0 = 0.55
    finals = []
    for rep in range(4000):
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture")
        for z in rng.binomial(1, p0, size=60):
            ep.update(z)
        finals.append(ep.e_value)
    assert np.mean(finals) < 1.15, f"mean e-value {np.mean(finals):.3f} should be ~<= 1"


def test_ville_false_certification():
    """Prob of ever crossing 1/alpha under the null <= alpha."""
    rng = np.random.default_rng(1)
    p0, alpha, n_rep = 0.55, 0.05, 3000
    crossed = 0
    for rep in range(n_rep):
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture", alpha=alpha)
        for z in rng.binomial(1, p0, size=256):
            ep.update(z)
            if ep.rejects():
                crossed += 1
                break
    rate = crossed / n_rep
    assert rate <= alpha + 0.01, f"false-cert rate {rate:.4f} > alpha {alpha}"


def test_revocation_null_control():
    """Under exact unlearning (p = 1/2) the revocation arm rarely fires."""
    rng = np.random.default_rng(2)
    alpha, n_rep = 0.05, 3000
    fired = 0
    for rep in range(n_rep):
        ep = MixtureEProcess(n_scores=3, m0=0.5, direction="above", alpha=alpha)
        zs = rng.binomial(1, 0.5, size=(128, 3))
        for row in zs:
            ep.update(row)
            if ep.rejects():
                fired += 1
                break
    rate = fired / n_rep
    assert rate <= alpha + 0.01, f"false-revocation rate {rate:.4f} > alpha {alpha}"


def test_symmetry_eprocess_null():
    """Magnitude-aware arm controls error under symmetric D."""
    rng = np.random.default_rng(3)
    alpha, n_rep = 0.05, 2000
    fired = 0
    for rep in range(n_rep):
        ep = SymmetryEProcess(alpha=alpha)
        for d in rng.standard_normal(128):
            ep.update(d, tie_break=rng.random())
            if ep.rejects():
                fired += 1
                break
    rate = fired / n_rep
    assert rate <= alpha + 0.01, f"symmetry false alarm {rate:.4f} > {alpha}"


def test_cs_coverage_uniform():
    """CS covers the true mean at every time with prob >= 1 - alpha."""
    rng = np.random.default_rng(4)
    alpha, n_rep, p = 0.05, 1500, 0.6
    misses = 0
    for rep in range(n_rep):
        cs = BettingCS(alpha=alpha, grid=401)
        ok = True
        for z in rng.binomial(1, p, size=100):
            cs.update(z)
            if not (cs.lo <= p <= cs.hi):
                ok = False
                break
        misses += (not ok)
    rate = misses / n_rep
    assert rate <= alpha + 0.015, f"CS miss rate {rate:.4f} > alpha {alpha}"


def test_certificate_power():
    """Under exact unlearning the certificate arrives, and revocation
    fires under strong residual memorization."""
    rng = np.random.default_rng(5)
    cfg = VouchConfig(eps=0.10, alpha=0.05)
    # exact unlearning: p = 1/2.  Theorem 3: E[tau*] ~ log(1/alpha) / KL(1/2, 1/2 + eps/2)
    # ~ 600 pairs at eps=0.1; sampling noise makes single runs slow sometimes,
    # so require issuance in most replications.
    issued = 0
    for rep in range(10):
        v = VouchVerifier(["loss"], cfg)
        diffs = [{"loss": float(d)} for d in rng.standard_normal(2048)]
        issued += (v.run(diffs, shuffle_seed=rep).status == "ISSUED")
    assert issued >= 7, f"only {issued}/10 exact-unlearning runs certified"
    # memorized: D strongly positive -> revocation must fire
    v2 = VouchVerifier(["loss"], cfg)
    diffs2 = [{"loss": float(d)} for d in rng.standard_normal(1024) + 2.0]
    cert2 = v2.run(diffs2)
    assert cert2.status == "REVOKED", cert2.status
    assert 0 < cert2.t_revoked < 100, "revocation should fire fast under strong leakage"
    # over-forgetting (in-twin scores pushed BELOW ghosts) must also be caught
    v3 = VouchVerifier(["loss"], cfg)
    diffs3 = [{"loss": float(d)} for d in rng.standard_normal(1024) - 2.0]
    cert3 = v3.run(diffs3)
    assert cert3.status == "REVOKED", f"over-forgetting missed: {cert3.status}"


def test_manifest_commitment():
    man = PGCGenerator(seed=7).generate(m=32, wave=1)
    c = man.commitment()
    j = man.to_json()
    man2 = CanaryManifest.from_json(j)
    assert man2.verify(c)
    # tamper detection
    man2.pairs[0].coin ^= 1
    assert not man2.verify(c)
    # in/ghost twins partition the pair
    p = man.pairs[0]
    assert p.in_text != p.ghost_text
    texts = man.in_twin_texts_with_repetition()
    assert len(texts) == sum(q.repetition for q in man.pairs)


def _run_all():
    mod = sys.modules[__name__]
    fails = 0
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            try:
                getattr(mod, name)()
                print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                fails += 1
    if fails:
        raise SystemExit(f"{fails} test(s) failed")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
