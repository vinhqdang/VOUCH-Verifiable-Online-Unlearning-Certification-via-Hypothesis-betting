#!/usr/bin/env python3
"""Re-score saved benchmark verdicts from stored pair_diffs.

The end-to-end runners save the raw per-pair score differences, so any
verifier configuration can be replayed offline (scoring is the expensive
part; betting is free).  This script recomputes each run's verdicts under
a chosen revocation configuration and writes a *_rescored.json alongside.

Default: sign-based revocation only (use_magnitude_revocation=False), which
matches Theorem 1 exactly and is the well-behaved main-protocol arm.  The
magnitude-aware arm (VOUCH+) is powerful for sparse leakage (see the
simulation ablation) but bets aggressively enough to produce occasional
transient early crossings on near-null models, so it is reported separately
as an enhancement rather than in the headline benchmark tables.

Usage: python3 experiments/rescore.py results/lm_e2e_muse_gpt2_512.json [...]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.verify import VouchConfig, VouchVerifier


def rescore_run(run: dict, use_magnitude: bool) -> dict:
    eps = run.get("eps", 0.2)
    alpha = run.get("alpha", 0.05)
    out = dict(run)
    out["certs"] = {}
    for method, cert in run["certs"].items():
        diffs = cert.get("pair_diffs")
        if not diffs:
            out["certs"][method] = cert
            continue
        scores = cert.get("score_class") or list(diffs[0].keys())
        v = VouchVerifier(scores, VouchConfig(
            eps=eps, alpha=alpha, use_magnitude_revocation=use_magnitude))
        c = v.run(diffs, shuffle_seed=run["seed"], early_stop=True)
        rec = json.loads(c.to_json())
        # carry over the expensive-to-recompute annotations
        for k in ("mean_loss_diff", "utility_nll", "scoring_seconds"):
            if k in cert:
                rec[k] = cert[k]
        out["certs"][method] = rec
    return out


def main():
    paths = sys.argv[1:]
    use_magnitude = "--magnitude" in paths
    paths = [p for p in paths if not p.startswith("--")]
    for path in paths:
        runs = json.load(open(path))
        rescored = [rescore_run(r, use_magnitude) for r in runs]
        suffix = "_rescored_mag" if use_magnitude else "_rescored"
        out = path.replace(".json", suffix + ".json")
        with open(out, "w") as f:
            json.dump(rescored, f, indent=2, default=float)
        # summary
        order = ["none", "retrain", "ga", "grad_diff", "npo",
                 "npo_P1_relearn", "npo_P3_jailbreak"]
        print(f"=== {os.path.basename(path)} "
              f"(revocation: {'sign+magnitude' if use_magnitude else 'sign-only'}) ===")
        for m in order:
            cs = [r["certs"][m] for r in rescored if m in r["certs"]]
            if not cs:
                continue
            v = "/".join(c["status"][0] for c in cs)
            print(f"  {m:<16} {v}")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
