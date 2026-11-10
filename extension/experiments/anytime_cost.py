#!/usr/bin/env python3
"""The price of anytime validity in the sign model (extension, theory).

Model: signs Z_1, Z_2, ... i.i.d. Bernoulli(p), p = (1 + D)/2, tested against the
exact null p = 1/2 (a leaking model has D > 0). For one arm at level a:

  oracle bettor   constant stake lambda = D, wealth W_t = prod (1 + D(2Z_u - 1));
                  growth rate g = KL(p || 1/2); stops at tau = inf{W_t >= 1/a}.
  upper bound     E[tau] <= (log(1/a) + log(1 + D)) / KL          (Proposition 8)
  Wald bound      any level-a test that stops almost surely under p has
                  E[tau] >= log(1/a) / KL                          (Proposition 8)
  grid mixture    uniform mixture of K constant stakes: E[tau] <= (log(K/a) + log(1+l))/g(l)
  fixed sample    exact two-sided binomial sign test at level 2a, smallest n with
                  power >= 0.5, 0.8, 0.9.

The two-sided sequential test runs two arms at level a = alpha/2 each, so the fair
fixed-sample comparison is the two-sided test at level alpha.

Usage: python3 extension/experiments/anytime_cost.py
"""
import json
import math
import os

import numpy as np
from scipy.stats import binom

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALPHA = 0.05
DELTAS = [0.05, 0.1, 0.2, 0.3, 0.5]
GRID = [0.05 * k for k in range(1, 11)]       # the stakes of check_cohort_growth.py


def kl(d):
    p = (1 + d) / 2
    return p * math.log(2 * p) + (1 - p) * math.log(2 * (1 - p))


def growth(lam, d):
    p = (1 + d) / 2
    return p * math.log(1 + lam) + (1 - p) * math.log(1 - lam)


def upper_bound(d, a, k=1, lam=None):
    lam = d if lam is None else lam
    return (math.log(k / a) + math.log(1 + lam)) / growth(lam, d)


def wald_lower(d, a):
    return math.log(1 / a) / kl(d)


def simulate_tau(d, a, lams, rng, runs=2000, cap=2_000_000):
    """Stopping time of the uniform mixture of constant-stake bettors (K = len(lams))."""
    p = (1 + d) / 2
    lams = np.asarray(lams)
    out = []
    log_target = math.log(1 / a)
    for _ in range(runs):
        logw = np.zeros(len(lams))
        t = 0
        while t < cap:
            # draw a block of signs at once
            z = rng.random(4096) < p
            s = 2 * z.astype(float) - 1
            inc = np.log1p(np.outer(s, lams))
            cum = logw + np.cumsum(inc, axis=0)
            mix = np.log(np.mean(np.exp(cum), axis=1))
            hit = np.nonzero(mix >= log_target)[0]
            if len(hit):
                out.append(t + hit[0] + 1)
                break
            logw = cum[-1]
            t += 4096
    return np.array(out)


def fixed_n(d, alpha, power):
    """Smallest n whose exact two-sided sign test at level <= alpha has power >= `power`."""
    p = (1 + d) / 2
    n = 8
    while True:
        # exact rejection region: counts k with P_0(|K - n/2| >= |k - n/2|) <= alpha
        ks = np.arange(n + 1)
        pk0 = binom.pmf(ks, n, 0.5)
        dev = np.abs(ks - n / 2)
        tail = np.array([pk0[dev >= x].sum() for x in dev])
        reject = tail <= alpha
        if binom.pmf(ks, n, p)[reject].sum() >= power:
            return n
        n = int(n * 1.04) + 1


def main():
    rng = np.random.default_rng(0)
    a = ALPHA / 2
    rows = []
    print(f"alpha={ALPHA}, two arms at level {a}")
    print(f"{'D':>5s} {'KL':>8s} {'Wald':>8s} {'oracle<=':>9s} {'sim oracle':>10s} {'sim grid':>9s}"
          f" {'grid<=':>8s} {'n50':>6s} {'n80':>6s} {'n90':>6s}")
    for d in DELTAS:
        lam_best = min(GRID, key=lambda l: abs(l - d))
        sim_o = simulate_tau(d, a, [d], rng, runs=2000).mean()
        sim_g = simulate_tau(d, a, GRID, rng, runs=500).mean()
        row = {"delta": d, "kl": kl(d), "wald": wald_lower(d, a), "oracle_upper": upper_bound(d, a),
               "sim_oracle": float(sim_o), "sim_grid": float(sim_g),
               "grid_upper": upper_bound(d, a, k=len(GRID), lam=lam_best),
               "n50": fixed_n(d, ALPHA, 0.5), "n80": fixed_n(d, ALPHA, 0.8),
               "n90": fixed_n(d, ALPHA, 0.9)}
        rows.append(row)
        print(f"{d:5.2f} {row['kl']:8.5f} {row['wald']:8.0f} {row['oracle_upper']:9.0f} {sim_o:10.0f}"
              f" {sim_g:9.0f} {row['grid_upper']:8.0f} {row['n50']:6d} {row['n80']:6d} {row['n90']:6d}")
    json.dump({"alpha": ALPHA, "arm_level": a, "grid": GRID, "rows": rows},
              open(os.path.join(ROOT, "extension", "results", "anytime_cost.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
