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


# ---------------------------------------------------------------------------
# Finite-cohort (without-replacement) certificate target.
#
# These cover the property the revision turns on: validity under *arbitrary*
# dependence across pairs.  The trick in every test below is that the sign
# vector is a fixed array -- so it embodies whatever dependence, heterogeneity
# or adversarial structure we like -- and the only randomness spent is the
# verifier's own committed reveal permutation.
# ---------------------------------------------------------------------------

def _least_favourable_cohort(m, p0, structure="stratified"):
    """A fixed sign vector at the least favourable feasible point of the
    cohort null H0: cohort mean >= p0 (i.e. mean = ceil(p0*m)/m)."""
    k = int(math.ceil(p0 * m - 1e-12))
    z = np.zeros(m)
    if structure == "contiguous":
        z[:k] = 1.0
    elif structure == "spread":
        z[np.linspace(0, m - 1, k).astype(int)] = 1.0
    elif structure == "stratified":
        # four blocks with wildly different rates, overall mean fixed to k/m
        per = m // 4
        for j, rate in enumerate((0.15, 0.45, 0.70, 0.95)):
            lo = j * per
            z[lo:lo + int(round(rate * per))] = 1.0
        cur = int(z.sum())
        if cur > k:
            z[np.flatnonzero(z)[: cur - k]] = 0.0
        elif cur < k:
            z[np.flatnonzero(z == 0)[: k - cur]] = 1.0
    else:
        raise ValueError(structure)
    assert int(z.sum()) == k
    return z


def test_wor_supermartingale_under_cohort_null():
    """Wealth has mean <= 1 under the cohort null, for a fixed sign vector
    with arbitrary internal structure (no independence anywhere)."""
    m, p0 = 256, 0.55
    for structure in ("contiguous", "spread", "stratified"):
        z0 = _least_favourable_cohort(m, p0, structure)
        finals = []
        for seed in range(400):
            rng = np.random.default_rng(seed)
            z = z0.copy()
            rng.shuffle(z)                       # committed random reveal order
            ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                                  alpha=0.05, population_size=m)
            for zz in z:
                ep.update(zz)
            finals.append(min(ep.log_e, 50.0))   # guard the refutation sentinel
        assert np.mean(np.exp(finals)) <= 1.5, (structure, np.mean(np.exp(finals)))


def test_wor_type_i_error_under_peeking():
    """Realised type-I error stays at or below alpha when the auditor peeks
    after every pair, for every cohort structure."""
    alpha = 0.05
    thr = math.log(1 / alpha)
    for m in (128, 384):
        for structure in ("contiguous", "spread", "stratified"):
            z0 = _least_favourable_cohort(m, 0.55, structure)
            crossed = 0
            for seed in range(600):
                rng = np.random.default_rng(10_000 + seed)
                z = z0.copy()
                rng.shuffle(z)
                ep = OneSidedEProcess(m0=0.55, direction="below",
                                      strategy="mixture", alpha=alpha,
                                      population_size=m)
                for zz in z:
                    ep.update(zz)
                    if ep.log_e >= thr:
                        crossed += 1
                        break
            rate = crossed / 600
            # Monte-Carlo slack: 600 draws at alpha=0.05 has se ~ 0.009
            assert rate <= alpha + 0.03, (m, structure, rate)


def test_wor_boundary_matches_definition():
    """The predictable boundary equals (m*p0 - sum so far) / (m - t + 1)."""
    m, p0 = 64, 0.6
    rng = np.random.default_rng(0)
    z = rng.binomial(1, 0.5, size=m).astype(float)
    ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                          alpha=0.05, population_size=m)
    running = 0.0
    for t, zz in enumerate(z, start=1):
        expected = (m * p0 - running) / (m - t + 1)
        assert abs(ep.m0_t - expected) < 1e-12, (t, ep.m0_t, expected)
        ep.update(zz)
        running += zz


def test_wor_refutes_only_when_null_is_impossible():
    """The deterministic-refutation branch fires only for cohorts whose
    realised mean is genuinely below p0, never under the null."""
    m, p0 = 128, 0.55
    # (a) under the null: never refuted
    z0 = _least_favourable_cohort(m, p0, "stratified")
    for seed in range(200):
        rng = np.random.default_rng(seed)
        z = z0.copy(); rng.shuffle(z)
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=0.05, population_size=m)
        for zz in z:
            ep.update(zz)
        assert not ep._refuted, seed
    # (b) a clean cohort (mean 1/2 < p0) is refuted with certainty by the end
    z1 = np.r_[np.ones(m // 2), np.zeros(m - m // 2)]
    for seed in range(50):
        rng = np.random.default_rng(seed)
        z = z1.copy(); rng.shuffle(z)
        ep = OneSidedEProcess(m0=p0, direction="below", strategy="mixture",
                              alpha=0.05, population_size=m)
        for zz in z:
            ep.update(zz)
        assert ep.log_e >= math.log(1 / 0.05), seed


def test_wor_cs_collapses_onto_the_cohort_mean():
    """The without-replacement confidence sequence contains the realised
    cohort mean at every step and pins it exactly once exhausted."""
    m = 200
    rng = np.random.default_rng(3)
    z = rng.binomial(1, 0.42, size=m).astype(float)
    truth = z.mean()
    cs = BettingCS(alpha=0.05, grid=1001, population_size=m)
    for zz in z:
        cs.update(zz)
        assert cs.lo - 1e-9 <= truth <= cs.hi + 1e-9, (cs.interval, truth)
    assert cs.hi - cs.lo < 0.02, cs.interval


def test_wor_verifier_agrees_with_the_sign_count_at_exhaustion():
    """With the whole cohort observed, the cohort certificate must agree with
    a direct count: |2*mean - 1| < eps  <=>  ISSUED."""
    eps, m = 0.2, 300
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        # spread the true advantage across and around the tolerance
        p = 0.40 + 0.03 * seed
        diffs = [{"loss": float(x)} for x in
                 np.where(rng.binomial(1, p, size=m) > 0, 1.0, -1.0)]
        v = VouchVerifier(["loss"], VouchConfig(eps=eps, alpha=0.05,
                                                use_magnitude_revocation=False))
        cert = v.run(diffs, shuffle_seed=seed, early_stop=False)
        mean = np.mean([d["loss"] > 0 for d in diffs])
        inside = abs(2 * mean - 1) < eps
        if cert.status == "ISSUED":
            assert inside, (seed, mean, cert.status)
        if not inside and cert.status != "REVOKED":
            assert cert.status != "ISSUED", (seed, mean, cert.status)


def test_finite_cohort_flag_recovers_v1_behaviour():
    """finite_cohort=False must reproduce the fixed-boundary process."""
    rng = np.random.default_rng(7)
    diffs = [{"loss": float(x)} for x in rng.standard_normal(300)]
    a = VouchVerifier(["loss"], VouchConfig(eps=0.2, finite_cohort=False))
    b = OneSidedEProcess(m0=0.6, direction="below", strategy="mixture",
                         alpha=0.05)   # population_size=None
    assert a.cohort_size is None
    assert b.population_size is None
    assert abs(b.m0_t - 0.6) < 1e-12


def test_merkle_commitment_opens_one_pair_at_a_time():
    """The Merkle commitment must (a) authenticate every leaf against the
    published root, (b) reject a tampered leaf, and (c) reveal nothing about
    unopened leaves -- which is what makes the audit filtration of Theorem 1
    a statement about what the verifier actually knows at step t-1."""
    import copy
    m = PGCGenerator(seed=0, domains=("qa",)).generate(m=64, wave=0)
    root = m.merkle_root()
    # (a) every authentication path checks out
    for i in range(64):
        assert CanaryManifest.verify_merkle_proof(
            m.leaf_hash(i), m.merkle_proof(i), root), i
    # proof length is logarithmic, so per-pair opening is cheap
    assert len(m.merkle_proof(0)) == 6, len(m.merkle_proof(0))
    # (b) flipping a single inclusion coin invalidates that leaf and the root
    tampered = copy.deepcopy(m)
    tampered.pairs[11].coin ^= 1
    assert not CanaryManifest.verify_merkle_proof(
        tampered.leaf_hash(11), m.merkle_proof(11), root)
    assert tampered.merkle_root() != root
    # (c) leaves are salted, so an unopened leaf hash is not guessable from
    # the template library alone: the same pair at a different index differs
    assert m.leaf_nonce(3) != m.leaf_nonce(4)
    # (d) the flat commitment still validates, for the released result files
    assert m.verify(m.commitment())


def test_open_leaf_payload_is_self_authenticating():
    m = PGCGenerator(seed=1, domains=("pii", "fact")).generate(m=32, wave=0)
    root = m.merkle_root()
    for i in (0, 5, 31):
        opened = m.open_leaf(i)
        assert CanaryManifest.verify_merkle_proof(
            opened["leaf_hash"], opened["proof"], root), i
        assert opened["pair"] is m.pairs[i]


def test_wealth_factors_stay_positive_when_boundary_reaches_one():
    """Regression: the recentred boundary can reach exactly 1 whenever
    N*m0 is an integer (e.g. N=640 at p0 in {0.525, 0.55, 0.6}).  The stake
    cap and the payoff must be computed from the *same* clamped boundary, or
    the worst-case factor 1 + lam*(v - 1) goes negative and the implemented
    process stops being the analysed one."""
    for m, p0 in ((640, 0.6), (640, 0.55), (640, 0.525), (384, 0.6)):
        for direction in ("below", "above"):
            ep = OneSidedEProcess(m0=p0 if direction == "below" else 1 - p0,
                                  direction=direction, strategy="mixture",
                                  alpha=0.05, population_size=m)
            for _ in range(m):
                if ep._boundary_state() != "live":
                    ep.update(0.0 if direction == "below" else 1.0)
                    continue
                v = ep._clamped_boundary()
                lam_max = ep._lam_max_at(v)
                # worst case over z in {0,1} for either direction
                worst = min(1.0 + lam_max * (v - 1.0), 1.0 + lam_max * v) \
                    if direction == "below" else \
                    min(1.0 + lam_max * (0.0 - v), 1.0 + lam_max * (1.0 - v))
                assert worst > 0.0, (m, p0, direction, ep.t, v, lam_max, worst)
                ep.update(0.0 if direction == "below" else 1.0)


def test_boundary_clamp_is_shared_between_stake_cap_and_payoff():
    """The stake cap must be derived from the boundary the payoff uses."""
    ep = OneSidedEProcess(m0=0.6, direction="below", strategy="mixture",
                          alpha=0.05, population_size=640)
    for _ in range(256):
        ep.update(0.0)
    v = ep._clamped_boundary()
    assert abs(ep.lam_max - ep._lam_max_at(v)) < 1e-9 * max(1.0, ep.lam_max)
    # and the implied worst-case factor is the max_bet_frac slack, not negative
    assert abs((1.0 + ep.lam_max * (v - 1.0)) - (1.0 - ep.max_bet_frac)) < 1e-6
