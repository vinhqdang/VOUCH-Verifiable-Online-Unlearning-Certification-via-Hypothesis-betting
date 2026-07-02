#!/usr/bin/env python3
"""Offline re-analysis of end-to-end LM runs.

run_lm_e2e.py stores the raw per-pair score differences in its JSON output,
so any verifier variant can be re-run without touching the model again
(scoring is the expensive part; betting is free).  This script reports, for
each run and subject:

  * two-sided (default) vs one-sided certification;
  * VOUCH decisions vs fixed-n baselines on the same pairs
    (exact binomial, TOST equivalence, permutation) -- the key ablation
    separating the inference contribution from the protocol contribution;
  * verifier compute (pairs consumed at decision vs full cohort).

Usage: python3 experiments/reanalyze.py results/lm_e2e_tiny.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.baselines.fixed_n import (binom_test_rev, permutation_test,
                                     tost_equivalence)
from vouch.canaries import PGCGenerator
from vouch.verify import VouchConfig, VouchVerifier


def dose_response(runs) -> None:
    """Delta(r) per repetition stratum (Section 6.7): pair_diffs are stored
    in manifest order, and manifests regenerate deterministically from the
    run seed, so strata can be recovered offline."""
    print("\n--- dose-response: mean D_loss by repetition stratum ---")
    header_done = False
    for run in runs:
        man = PGCGenerator(seed=run["seed"]).generate(m=run["m_pairs"], wave=0)
        if man.commitment() != run["manifest_sha256"]:
            print(f"seed {run['seed']}: manifest mismatch, skipping")
            continue
        reps = np.array([p.repetition for p in man.pairs])
        for method, cert in run["certs"].items():
            diffs = cert.get("pair_diffs")
            if not diffs or len(diffs) != len(reps):
                continue
            d = np.array([x["loss"] for x in diffs])
            strata = sorted(set(reps.tolist()))
            if not header_done:
                print("seed  method        " +
                      "  ".join(f"r={r}".rjust(8) for r in strata))
                header_done = True
            row = "  ".join(f"{d[reps == r].mean():8.3f}" for r in strata)
            print(f"{run['seed']:<5} {method:<13} {row}")


def reanalyze(path: str) -> None:
    with open(path) as f:
        runs = json.load(f)
    rows = []
    for run in runs:
        seed = run["seed"]
        eps, alpha = run["eps"], run["alpha"]
        for method, cert in run["certs"].items():
            diffs = cert.get("pair_diffs")
            if not diffs:
                continue
            scores = cert["score_class"]
            # two-sided (shipped default)
            v2 = VouchVerifier(scores, VouchConfig(eps=eps, alpha=alpha, two_sided=True))
            c2 = v2.run(diffs, shuffle_seed=seed)
            # one-sided (design-doc semantics)
            v1 = VouchVerifier(scores, VouchConfig(eps=eps, alpha=alpha, two_sided=False))
            c1 = v1.run(diffs, shuffle_seed=seed)
            # fixed-n baselines on the loss-score signs over the FULL cohort
            d_loss = np.array([d["loss"] for d in diffs])
            z = (d_loss > 0).astype(float)
            p_rev = binom_test_rev(z)
            p_tost = tost_equivalence(z, eps)
            p_perm = permutation_test(d_loss, seed=seed)
            rows.append({
                "seed": seed, "method": method,
                "vouch_two_sided": c2.status, "t2": c2.t_stop,
                "t2_revoked": c2.t_revoked,
                "vouch_one_sided": c1.status,
                "delta_upper": round(c2.delta_upper, 3),
                "binom_rev_p": round(p_rev, 4),
                "tost_equiv_p": round(p_tost, 4),
                "perm_p": round(p_perm, 4),
                "mean_D_loss": round(float(d_loss.mean()), 3),
            })
    if not rows:
        print("no pair_diffs found in", path)
        return
    hdr = list(rows[0].keys())
    w = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in hdr}
    print("  ".join(h.ljust(w[h]) for h in hdr))
    for r in rows:
        print("  ".join(str(r[h]).ljust(w[h]) for h in hdr))
    out = path.replace(".json", "_reanalysis.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[saved] {out}")
    dose_response(runs)


if __name__ == "__main__":
    for p in sys.argv[1:] or [os.path.join(os.path.dirname(__file__), "..",
                                           "results", "lm_e2e_tiny.json")]:
        print(f"=== {p} ===")
        reanalyze(p)
