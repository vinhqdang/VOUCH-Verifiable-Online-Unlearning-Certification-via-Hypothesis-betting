#!/usr/bin/env python3
"""Analyse a run written by run_baselines.py (extension, partial unlearning).

Analysis over the whole grid of strengths (no strength window is selected):

  advantages   realised cohort advantage of every VOUCH score, per subject,
               mean and standard error over seeds
  retention    for each unlearning subject and seed, advantage after unlearning
               divided by the same seed's advantage before unlearning ("none").
               These per-seed ratios are unstable when a seed's pre-unlearning
               advantage is near zero; the manuscript therefore normalises by
               the mean pre-unlearning advantage over seeds instead
               (make_tables_ext.py), and reports this per-seed version alongside.
  test         like-for-like comparison of the base-calibrated scores:
               paraphrase (belief_cal) retention minus literal (literal_cal)
               retention, averaged over ALL unlearning strengths of the grid
               within each seed, then a Wilcoxon signed-rank test over seeds
               (one value per seed, so cells sharing a seed are not treated as
               independent)
  secondary    the same for the other score pairings, reported for transparency
  audit rules  fraction of runs (seeds x K reveal orders) ending REVOKED and
               ISSUED under the literal-only, union and mixture rules of
               union_audit.py, including the retrain control (false alarms)

Usage: python3 extension/experiments/analyze_runs.py extension/results/runA_semantic_literal.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from union_audit import cell  # noqa: E402

SCORES = ["literal", "literal_cal", "belief", "belief_cal", "belief_max", "belief_max_cal"]
PAIRINGS = [("literal_cal", "belief_cal"), ("literal", "belief"),
            ("literal", "belief_cal"), ("literal_cal", "belief_max_cal")]


def adv(diffs, k):
    return 2 * np.mean([d[k] > 0 for d in diffs]) - 1


def order(names):
    rest = sorted(set(names) - {"none", "retrain"},
                  key=lambda s: (s.split("@")[0], int(s.split("@")[1])))
    return [n for n in ["none", "retrain"] if n in names] + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [r for r in json.load(open(args.path))
            if "none" in r["subjects"] and "retrain" in r["subjects"]]
    names = order({n for r in runs for n in r["subjects"]})
    names = [n for n in names if all(n in r["subjects"] for r in runs)]
    A = {(r["seed"], n): {k: adv(r["subjects"][n]["pair_diffs"], k) for k in SCORES}
         for r in runs for n in names}
    seeds = [r["seed"] for r in runs]
    unl = [n for n in names if "@" in n]
    out = {"source": os.path.relpath(args.path, ROOT), "seeds": seeds, "subjects": names}

    # advantages
    out["advantage"] = {n: {k: [float(np.mean([A[(s, n)][k] for s in seeds])),
                                float(np.std([A[(s, n)][k] for s in seeds], ddof=1) / np.sqrt(len(seeds)))]
                            for k in SCORES} for n in names}

    # retention and per-seed tests
    def ret(s, n, k):
        return A[(s, n)][k] / A[(s, "none")][k]
    out["retention"] = {n: {k: float(np.mean([ret(s, n, k) for s in seeds])) for k in SCORES}
                        for n in unl}
    tests = {}
    for a, b in PAIRINGS:
        per_seed = np.array([np.mean([ret(s, n, b) - ret(s, n, a) for n in unl]) for s in seeds])
        per_strength = {n: float(wilcoxon([ret(s, n, b) - ret(s, n, a) for s in seeds]).pvalue)
                        for n in unl}
        tests[f"{b}-{a}"] = {"per_seed": per_seed.tolist(), "mean": float(per_seed.mean()),
                             "se": float(per_seed.std(ddof=1) / np.sqrt(len(seeds))),
                             "positive": int((per_seed > 0).sum()), "n": len(seeds),
                             "p": float(wilcoxon(per_seed).pvalue),
                             "per_strength_p": per_strength,
                             "base_ratio": float(np.mean([A[(s, "none")][b] / A[(s, "none")][a]
                                                          for s in seeds]))}
    out["tests"] = tests

    # audit rules
    keys = [(r["seed"], n) for r in runs for n in names]
    with Pool(args.procs) as pool:
        res = pool.map(cell, [(r["subjects"][n]["pair_diffs"], 0.05, 0.2, args.k)
                              for r in runs for n in names])
    rules = {}
    for n in names:
        idx = [i for i, (_, m) in enumerate(keys) if m == n]
        rules[n] = {f"{st.lower()}_{u}": float(np.mean([x == st for i in idx for x in res[i][u]]))
                    for u in ("literal", "union", "mixture") for st in ("REVOKED", "ISSUED")}
        rules[n]["literal_issued_union_revoked"] = float(np.mean(
            [a == "ISSUED" and b == "REVOKED" for i in idx
             for a, b in zip(res[i]["literal"], res[i]["union"])]))
        # seed-level: in how many seeds does the majority of orders show that pattern
        rules[n]["seeds_majority_literal_issued_union_revoked"] = int(sum(
            np.mean([a == "ISSUED" and b == "REVOKED"
                     for a, b in zip(res[i]["literal"], res[i]["union"])]) > 0.5 for i in idx))
    out["rules"] = rules
    out["k"] = args.k

    print(f"{args.path}: seeds {seeds}")
    print(f"{'subject':9s} " + " ".join(f"{k:>14s}" for k in SCORES))
    for n in names:
        print(f"{n:9s} " + " ".join(f"{out['advantage'][n][k][0]:+.3f}±{out['advantage'][n][k][1]:.3f}"
                                    for k in SCORES))
    for k, t in tests.items():
        print(f"retention gap {k:28s} mean {t['mean']:+.3f}±{t['se']:.3f} pos {t['positive']}/{t['n']} "
              f"p={t['p']:.4f} base ratio {t['base_ratio']:.2f}")
    print(f"{'subject':9s} revoked lit/union/mix   issued lit/union/mix   lit-iss&union-rev (seeds)")
    for n in names:
        r = rules[n]
        print(f"{n:9s} {r['revoked_literal']:.2f} {r['revoked_union']:.2f} {r['revoked_mixture']:.2f}"
              f"        {r['issued_literal']:.2f} {r['issued_union']:.2f} {r['issued_mixture']:.2f}"
              f"        {r['literal_issued_union_revoked']:.2f} ({r['seeds_majority_literal_issued_union_revoked']})")
    json.dump(out, open(args.out or args.path.replace(".json", "_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
