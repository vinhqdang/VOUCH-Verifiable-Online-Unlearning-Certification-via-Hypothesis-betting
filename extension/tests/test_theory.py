"""Numerical checks of the extension's new propositions (no model weights needed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from anytime_cost import GRID, growth, kl, simulate_tau, upper_bound, wald_lower  # noqa: E402


def np_power(p, q, eps):
    """sup E_q[phi] subject to E_p[phi] <= eps, phi in [0,1]: fractional Neyman-Pearson."""
    order = np.argsort(-(q / np.maximum(p, 1e-300)))
    budget, val = eps, 0.0
    for i in order:
        if p[i] <= 1e-300:
            val += q[i]
            continue
        take = min(1.0, budget / p[i])
        val += take * q[i]
        budget -= take * p[i]
        if budget <= 1e-15:
            break
    return val


def test_np_sandwich_and_extremes():
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = 7
        p = rng.dirichlet(np.ones(k))
        q = rng.dirichlet(np.ones(k) * rng.uniform(0.3, 3))
        tv = 0.5 * np.abs(p - q).sum()
        a = p[q > p].sum()                      # P(B*) for B* = {q > p}
        for eps in (0.02, 0.1, 0.2, 0.5):
            beta = np_power(p, q, eps)
            assert eps - 1e-9 <= beta <= min(1.0, eps + tv) + 1e-9
            if eps >= a:
                assert beta >= a + tv - 1e-9    # B* itself is allowed
            else:
                assert beta >= eps * (1 + tv / a) - 1e-9   # scaled B*
    # identical distributions: no gap; mutually singular: everything leaks
    p = np.array([0.5, 0.5, 0.0, 0.0])
    assert abs(np_power(p, p, 0.2) - 0.2) < 1e-12
    q = np.array([0.0, 0.0, 0.5, 0.5])
    assert np_power(p, q, 0.0) == 1.0


def test_wald_and_oracle_bounds():
    rng = np.random.default_rng(2)
    a = 0.025
    for d in (0.2, 0.4):
        tau = simulate_tau(d, a, [d], rng, runs=1500)
        se = tau.std(ddof=1) / math.sqrt(len(tau))
        assert tau.mean() >= wald_lower(d, a) - 4 * se
        assert tau.mean() <= upper_bound(d, a) + 4 * se


def test_growth_rate_is_kl_at_oracle_stake():
    for d in (0.05, 0.2, 0.6):
        assert abs(growth(d, d) - kl(d)) < 1e-12
        assert all(growth(l, d) <= kl(d) + 1e-12 for l in GRID if l < 1)


def test_average_dominates_bonferroni_union():
    rng = np.random.default_rng(3)
    alpha = 0.05
    for _ in range(10000):
        e = rng.exponential(1.0, size=2) * rng.choice([1, 10, 100], size=2)
        union = (e >= 2 / alpha).any()
        mixed = e.mean() >= 1 / alpha
        assert (not union) or mixed
        w = rng.dirichlet([1, 1])
        weighted = (w * e).sum() >= 1 / alpha
        bonf = (e >= 1 / (w * alpha)).any()
        assert (not bonf) or weighted
