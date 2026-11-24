#!/usr/bin/env python3
"""Offline VOUCH verdicts for the end-to-end selective-provider runs.

For every (seed, subject) of run_selective.py and K seeded reveal orders, runs the
VOUCH verifier at (eps, alpha) on
  canary   the auditor's cohort (the certificate the provider must pass)
  probe    the organic-matched probe pairs (the organic advantage Delta^o)
with the declared class {literal, literal_cal}. Reports per subject the mean
advantages (sign statistic) with standard errors over seeds, the fraction of
(seed, order) runs ISSUED and REVOKED for each group, the number of seeds whose
majority verdict is ISSUED / REVOKED, and the membership-inference AUC of the
organic forget biographies against the organic holdout (loss score).

Usage: python3 extension/experiments/analyze_selective.py extension/results/selective_e2e.json \
           --out extension/results/selective_e2e_analysis.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from vouch.verify import VouchConfig, VouchVerifier  # noqa: E402

SCORES = ["literal", "literal_cal"]


def cell(job):
    subj, group, alpha, eps, k = job
    diffs = [{"literal": a, "literal_cal": b}
             for a, b in zip(subj[group + "_literal"], subj[group + "_literal_cal"])]
    out = []
    for j in range(k):
        v = VouchVerifier(SCORES, VouchConfig(eps=eps, alpha=alpha))
        out.append(v.run(diffs, shuffle_seed=1000 + j, reveal_source="seeded",
                         early_stop=True).status)
    return out


def auc(members, nonmembers):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))]
    return float(roc_auc_score(y, -np.r_[members, nonmembers]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = json.load(open(args.path))
    names = []
    for r in runs:
        for nm in r["subjects"]:
            if nm not in names:
                names.append(nm)
    jobs, keys = [], []
    for r in runs:
        for nm, s in r["subjects"].items():
            for g in ("canary", "probe"):
                jobs.append((s, g, args.alpha, args.eps, args.k))
                keys.append((r["seed"], nm, g))
    with Pool(args.procs) as pool:
        res = dict(zip(keys, pool.map(cell, jobs)))

    rows = []
    for nm in names:
        seeds = [r for r in runs if nm in r["subjects"]]
        row = {"subject": nm, "n_seeds": len(seeds)}
        for g in ("canary", "probe"):
            for sc in SCORES:
                a = [np.mean(np.sign(r["subjects"][nm][f"{g}_{sc}"])) for r in seeds]
                row[f"{g}_adv_{sc}"] = [float(np.mean(a)),
                                        float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0]
            st = [res[(r["seed"], nm, g)] for r in seeds]
            flat = [x for s in st for x in s]
            for v in ("ISSUED", "REVOKED"):
                row[f"{g}_{v.lower()}"] = float(np.mean([x == v for x in flat]))
                row[f"{g}_seeds_majority_{v.lower()}"] = int(sum(
                    np.mean([x == v for x in s]) > 0.5 for s in st))
        row["organic_auc_loss"] = float(np.mean([auc(r["subjects"][nm]["forget_bio_loss"],
                                                     r["subjects"][nm]["holdout_loss"])
                                                 for r in seeds]))
        rows.append(row)
        print(f"{nm:14s} canary adv={row['canary_adv_literal'][0]:+.3f} "
              f"iss={row['canary_issued']:.2f} rev={row['canary_revoked']:.2f} | "
              f"probe adv={row['probe_adv_literal'][0]:+.3f} "
              f"iss={row['probe_issued']:.2f} rev={row['probe_revoked']:.2f} | "
              f"organic AUC={row['organic_auc_loss']:.3f}")
    # pooled sign tests over all seeds' pairs, and paired per-seed comparisons
    from scipy.stats import binomtest, wilcoxon
    pooled = {}
    for nm in names:
        seeds = [r for r in runs if nm in r["subjects"]]
        for g in ("canary", "probe"):
            z = np.concatenate([np.array(r["subjects"][nm][g + "_literal"]) > 0 for r in seeds])
            k, n = int(z.sum()), len(z)
            pooled[f"{nm}/{g}"] = {"adv": 2 * k / n - 1, "n": n,
                                   "p_two_sided": float(binomtest(k, n).pvalue)}
    adv = lambda r, nm, g: float(np.mean(np.sign(r["subjects"][nm][g + "_literal"])))
    paired = {}
    for a_, b_ in (("selective@150", "honest@150"), ("selective@75", "honest@75"),
                   ("selective@150", "retrain"), ("selective@75", "retrain"),
                   ("selective@150", "none")):
        both = [r for r in runs if a_ in r["subjects"] and b_ in r["subjects"]]
        if len(both) < 2:
            continue
        d = [adv(r, a_, "probe") - adv(r, b_, "probe") for r in both]
        paired[f"{a_} - {b_}"] = {"per_seed": d, "mean": float(np.mean(d)),
                                  "positive": int(sum(x > 0 for x in d)), "n": len(d),
                                  "p_wilcoxon": float(wilcoxon(d).pvalue)}
    for k_, v in pooled.items():
        print(f"pooled {k_:22s} adv={v['adv']:+.3f} n={v['n']} p={v['p_two_sided']:.1e}")
    for k_, v in paired.items():
        print(f"paired probe {k_:26s} mean={v['mean']:+.3f} pos={v['positive']}/{v['n']} "
              f"p={v['p_wilcoxon']:.3f}")
    sel = {r["seed"]: r.get("selection") for r in runs}
    json.dump({"source": args.path, "eps": args.eps, "alpha": args.alpha, "k": args.k,
               "selection": sel, "rows": rows,
               "pooled": pooled, "paired": paired}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
