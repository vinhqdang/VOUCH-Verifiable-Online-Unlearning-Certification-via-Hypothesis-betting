#!/usr/bin/env python3
"""Replay every saved benchmark audit under independent reveal permutations.

Theorem 2 needs the reveal order to be independent of the realised signs. The
released runs used a permutation seeded from the run seed (reveal_source =
"seeded"), which does not meet that hypothesis. The raw per-pair scores are
saved, so the whole verification layer can be replayed with permutations drawn
independently of the data and no retraining. For each (run, subject, tolerance)
cell this script compares the original seeded verdict with K independent
permutations (cohort target) and reports how often the verdict changes.

Usage: python3 experiments/replay_reveal_orders.py [--k 50]
Writes results/reveal_replay.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vouch.verify import VouchConfig, VouchVerifier

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TAGS = ["tofu_gpt2", "tofu_pythia160m", "tofu_phi-1_5",
        "muse_gpt2_512", "muse_pythia160m", "muse_phi-1_5",
        "tofu_gpt2_rmu", "tofu_pythia_rmu"]
EPS = (0.2, 0.1, 0.05)
ALPHA = 0.05


def runs_for(tag):
    for name in (f"lm_e2e_{tag}", f"lm_e2e_{tag}_partial"):
        p = os.path.join(RES, name + ".json")
        if os.path.exists(p):
            d = json.load(open(p))
            return d if isinstance(d, list) else [d]
    return []


def verdict(diffs, names, eps, seed):
    v = VouchVerifier(names, VouchConfig(eps=eps, alpha=ALPHA,
                                         use_magnitude_revocation=False,
                                         finite_cohort=True))
    return v.run(diffs, shuffle_seed=seed, early_stop=True).status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(20261024)       # independent of every run seed and score
    out = {"k": args.k, "cells": [], "by_tag": {}}
    for tag in TAGS:
        n = chg = unstable = 0
        flips = {}
        for run in runs_for(tag):
            for m, c in run.get("certs", {}).items():
                diffs = c.get("pair_diffs")
                if not diffs:
                    continue
                names = list(diffs[0].keys())
                for eps in EPS:
                    orig = verdict(diffs, names, eps, run["seed"])
                    seeds = rng.integers(0, 2**62, size=args.k)
                    rep = [verdict(diffs, names, eps, int(s)) for s in seeds]
                    maj = max(set(rep), key=rep.count)
                    frac_same = float(np.mean([r == orig for r in rep]))
                    n += 1
                    chg += int(maj != orig)
                    unstable += int(len(set(rep)) > 1)
                    out["cells"].append({"tag": tag, "seed": run["seed"], "subject": m,
                                         "eps": eps, "seeded": orig,
                                         "replay": {s: rep.count(s) for s in set(rep)},
                                         "frac_same_as_seeded": frac_same,
                                         "majority_differs": maj != orig})
        out["by_tag"][tag] = {"cells": n, "majority_differs": chg,
                              "order_sensitive": unstable}
        print(f"{tag:18s} cells={n:3d} majority verdict differs from seeded: {chg:3d}  "
              f"any order-sensitivity: {unstable:3d}", flush=True)
    tot = {k: sum(v[k] for v in out["by_tag"].values()) for k in ("cells", "majority_differs", "order_sensitive")}
    out["total"] = tot
    print("TOTAL", tot)
    json.dump(out, open(os.path.join(RES, "reveal_replay.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
