#!/usr/bin/env python3
"""Turn the shadow-model runs of run_lira.py into the LiRA attack's verdict.

Reads results/lira_tiny.json and reports, per subject:

  * the realised cohort advantage the *declared class* achieves, which is
    just the sign of the target model's own score difference (this is
    s_loss restricted to the identity frame, i.e. what VOUCH itself sees);
  * the realised cohort advantage the *LiRA* statistic achieves, using the
    per-twin IN/OUT calibration built from the shadows;
  * an anytime-valid upper confidence bound on the LiRA advantage, from the
    same betting confidence sequence the certificate uses, so the number is
    comparable with the ``mean CS U_t`` column of the out-of-class table;
  * whether the LiRA advantage exceeds each tolerance, against the exact
    binomial null for a clean cohort of the same size (the convention
    Section 5.11 already uses).

Usage:  python experiments/analyze_lira.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.verify.betting import BettingCS

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def lira_lambda(phi_target, shadow_phi, shadow_coins):
    """Per-twin LiRA log-likelihood ratio with a globally pooled variance.

    shadow_phi[s][p][t] is twin t of pair p under shadow s; twin t was IN
    shadow s exactly when shadow_coins[s][p] == t.
    """
    S = len(shadow_phi)
    P = len(phi_target)
    mu_in = np.full((P, 2), np.nan)
    mu_out = np.full((P, 2), np.nan)
    var_pool = []
    for p in range(P):
        for t in (0, 1):
            ins = [shadow_phi[s][p][t] for s in range(S) if shadow_coins[s][p] == t]
            outs = [shadow_phi[s][p][t] for s in range(S) if shadow_coins[s][p] != t]
            if ins:
                mu_in[p, t] = np.mean(ins)
                if len(ins) > 1:
                    var_pool.append(np.var(ins, ddof=1))
            if outs:
                mu_out[p, t] = np.mean(outs)
                if len(outs) > 1:
                    var_pool.append(np.var(outs, ddof=1))
    sigma2 = float(np.mean(var_pool)) if var_pool else 1.0
    sigma2 = max(sigma2, 1e-12)
    phi = np.asarray(phi_target, dtype=float)
    # log N(phi; mu_in, s2) - log N(phi; mu_out, s2)
    lam = ((phi - mu_out) ** 2 - (phi - mu_in) ** 2) / (2.0 * sigma2)
    return lam, sigma2, mu_in, mu_out


def cs_upper(signs, alpha=0.05):
    """Anytime upper confidence bound on the cohort advantage."""
    cs = BettingCS(alpha=alpha, population_size=len(signs))
    for z in signs:
        cs.update(float(z))
    return float(cs.advantage_interval[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="lira_tiny.json")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="lira_analysis.json")
    args = ap.parse_args()

    d = json.load(open(os.path.join(RESULTS, args.infile)))
    coins = d["target_coins"]
    S = len(d["shadow_coins"])
    P = len(coins)
    print(f"shadows={S}  pairs={P}  subjects={list(d['target_phi'])}\n")

    report = {"shadows": S, "pairs": P, "subjects": {}}
    for subj, phi_t in d["target_phi"].items():
        if subj not in d["shadow_phi"] or not d["shadow_phi"][subj]:
            continue
        lam, sigma2, mu_in, mu_out = lira_lambda(
            phi_t, d["shadow_phi"][subj], d["shadow_coins"])
        phi = np.asarray(phi_t, dtype=float)

        # A twin needs at least one IN and one OUT shadow for its calibration
        # to exist; with S shadows that fails with probability 2^-S per twin,
        # so this is defensive rather than load-bearing.  Pairs whose twins
        # are not both calibrated are dropped, and the effective cohort size
        # is reported alongside the advantage it produced.
        usable = np.array([np.isfinite(lam[p]).all() for p in range(P)])
        idx = np.flatnonzero(usable)
        if idx.size == 0:
            print(f"{subj}: no calibrated pairs (need >= 2 shadows)"); continue

        # in-class: the target model's own score difference (what VOUCH sees)
        z_in_class = np.array([float(phi[p, coins[p]] > phi[p, 1 - coins[p]])
                               for p in idx])
        # LiRA: which twin does the calibrated likelihood ratio prefer?
        z_lira = np.array([float(lam[p, coins[p]] > lam[p, 1 - coins[p]])
                           for p in idx])

        d_in, d_li = 2 * z_in_class.mean() - 1, 2 * z_lira.mean() - 1
        u_in, u_li = cs_upper(z_in_class, args.alpha), cs_upper(z_lira, args.alpha)

        def delta_ci(z):
            """Clopper-Pearson interval on the sign rate, mapped to Delta.
            The betting CS collapses onto the realised value once the cohort
            is exhausted (Theorem 2), so it carries no width here; this is
            the estimate's sampling uncertainty over cohorts of this size."""
            k, n = int(z.sum()), len(z)
            a = 1 - 0.95
            lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
            hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
            return [2 * lo - 1, 2 * hi - 1]

        row = {"delta_in_class": d_in, "delta_lira": d_li,
               "delta_in_class_ci": delta_ci(z_in_class),
               "delta_lira_ci": delta_ci(z_lira),
               "cs_upper_in_class": u_in, "cs_upper_lira": u_li,
               "sigma2": sigma2, "pairs_used": int(idx.size),
               "agreement": float(np.mean(z_in_class == z_lira))}
        n_eff = int(idx.size)
        for eps in args.eps:
            # exact binomial null: P(|2*Binom(n,1/2)/n - 1| > eps) for a clean cohort
            k_hi = int(np.floor(n_eff * (1 + eps) / 2))
            null = float(2 * stats.binom.sf(k_hi, n_eff, 0.5))
            row[f"exceeds_eps={eps}"] = bool(abs(d_li) > eps)
            row[f"in_class_exceeds_eps={eps}"] = bool(abs(d_in) > eps)
            row[f"chance_rate_eps={eps}"] = null
        report["subjects"][subj] = row
        ci = row["delta_lira_ci"]
        print(f"{subj:9s} in-class Delta={d_in:+.3f}   "
              f"LiRA Delta={d_li:+.3f} 95% CI [{ci[0]:+.3f},{ci[1]:+.3f}]   "
              f"agree={row['agreement']:.3f}  "
              + "  ".join(f"eps={e}:{'EXCEEDS' if row[f'exceeds_eps={e}'] else 'ok'}"
                          for e in args.eps))

    with open(os.path.join(RESULTS, args.out), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\n[saved] results/{args.out}")


if __name__ == "__main__":
    main()
