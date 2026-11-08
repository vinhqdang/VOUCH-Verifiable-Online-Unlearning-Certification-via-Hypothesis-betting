#!/usr/bin/env python3
"""Numerical check of the finite-population certification-time bound.

Upper certificate arm of the without-replacement process: a committed cohort of
m signs with j ones (realised mean p_hat = j/m < p0), revealed in a uniformly
random order, bet against the recentred boundary
    p0(t) = (m p0 - S_{t-1}) / (m - t + 1)
by a uniform mixture of K constant-stake bettors with stakes in (0, 1/2].
The null is refuted outright once p0(t) > 1.

The theorem bounds, for every t,
    P(tau > t) <= min_lambda exp(-x^2 / (2 (V_t(lambda) + R_lambda x / 3))),
    x = (Psi_t(lambda) - log(K/alpha))_+,  V_t(lambda) = lambda^2 (t/4 + G_t) / (1-lambda)^2,
    Psi_t(lambda) = lambda A_t - lambda^2 (t/4 + G_t) / (2 (1-lambda)^2),
    A_t = m delta (H_m - H_{m-t}),  G_t = m^2 delta^2 sum_{k=m-t+1}^{m} k^{-2},
    R_lambda = log((1+lambda)/(1-lambda)),  delta = p0 - p_hat.
This script compares the bound with the empirical tail and with the same bettor
run against the fixed boundary p0 on i.i.d. Bern(p_hat) signs.

Usage: python3 extension/experiments/check_cohort_growth.py
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
LAMBDAS = np.arange(0.05, 0.501, 0.05)
ALPHA = 0.05


def run_cohort(z, p0, lambdas, alpha, rng):
    m = len(z)
    order = rng.permutation(m)
    logw = np.zeros(len(lambdas))
    s = 0
    thr = np.log(1 / alpha)
    for t in range(1, m + 1):
        b = (m * p0 - s) / (m - t + 1)
        if b > 1:
            return t                       # refutation
        zt = z[order[t - 1]]
        logw += np.log1p(lambdas * (b - zt))
        if np.log(np.mean(np.exp(logw - logw.max()))) + logw.max() >= thr:
            return t
        s += zt
    return m + 1


def run_fixed(p_hat, p0, m, lambdas, alpha, rng):
    logw = np.zeros(len(lambdas))
    thr = np.log(1 / alpha)
    for t in range(1, m + 1):
        zt = rng.random() < p_hat
        logw += np.log1p(lambdas * (p0 - zt))
        if np.log(np.mean(np.exp(logw - logw.max()))) + logw.max() >= thr:
            return t
    return m + 1


def bound_tail(t, m, delta, lambdas, alpha):
    k = np.arange(m - t + 1, m + 1)
    a = m * delta * np.sum(1.0 / k)
    g = (m * delta) ** 2 * np.sum(1.0 / k ** 2)
    best = 1.0
    for lam in lambdas:
        psi = lam * a - lam ** 2 * (t / 4 + g) / (2 * (1 - lam) ** 2)
        x = psi - np.log(len(lambdas) / alpha)
        if x <= 0:
            continue
        r = np.log((1 + lam) / (1 - lam))          # lower range of the increments
        v = lam ** 2 * (t / 4 + g) / (1 - lam) ** 2  # conditional-variance budget
        best = min(best, float(np.exp(-x * x / (2 * (v + r * x / 3)))))
    return best


def main():
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    report = []
    for m, eps, p_hat in ((512, 0.2, 0.5), (1024, 0.2, 0.5), (1024, 0.1, 0.5), (2048, 0.1, 0.5)):
        p0 = 0.5 + eps / 2
        j = int(round(p_hat * m))
        delta = p0 - j / m
        z = np.array([1] * j + [0] * (m - j))
        tc = np.array([run_cohort(z, p0, LAMBDAS, ALPHA, rng) for _ in range(2000)])
        tf = np.array([run_fixed(j / m, p0, m, LAMBDAS, ALPHA, rng) for _ in range(2000)])
        grid = [t for t in range(10, m + 1, max(1, m // 32))]
        rows = []
        violations = 0
        for t in grid:
            emp = float(np.mean(tc > t))
            bd = bound_tail(t, m, delta, LAMBDAS, ALPHA)
            # allow Monte-Carlo slack of three standard errors
            se = np.sqrt(max(emp * (1 - emp), 1e-4) / len(tc))
            if emp > bd + 3 * se:
                violations += 1
            rows.append({"t": t, "empirical_tail": emp, "bound": bd})
        t_bound_half = next((r["t"] for r in rows if r["bound"] <= 0.5), None)
        rec = {"m": m, "eps": eps, "p_hat": j / m, "delta": delta,
               "median_tau_cohort": float(np.median(tc)),
               "median_tau_fixed_boundary": float(np.median(np.minimum(tf, m + 1))),
               "frac_fixed_not_done_by_m": float(np.mean(tf > m)),
               "first_t_bound_le_half": t_bound_half,
               "bound_violations": violations, "grid_points": len(grid), "tail": rows}
        report.append(rec)
        print(f"m={m:5d} eps={eps} delta={delta:.3f}  median tau: cohort={rec['median_tau_cohort']:.0f}"
              f"  fixed-boundary={rec['median_tau_fixed_boundary']:.0f}"
              f"  bound<=1/2 from t={t_bound_half}  violations={violations}/{len(grid)}", flush=True)
    with open(os.path.join(OUT, "cohort_growth_check.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
