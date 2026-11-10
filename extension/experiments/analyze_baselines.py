#!/usr/bin/env python3
"""Compare membership-inference audits with VOUCH (extension, baselines).

Input: the per-record features written by run_baselines.py. For every subject
and seed it applies the audit practitioners run today, a membership-inference
AUC with a bootstrap confidence interval (the PrivLeak protocol of MUSE uses the
same AUC; we report the test form):

  member scores     -feature on the in-twin records (literal statement, or the
                    paraphrased questions never trained)
  non-member scores -feature on either
                      matched    the ghost twins (same generator, never trained)
                      unmatched  organic biographies never trained, the usual
                                 holdout set of an unlearning benchmark
  verdict           LEAK if the 95% bootstrap interval of the AUC excludes 0.5,
                    PASS otherwise

and VOUCH on the same runs, at eps = 0.2 and alpha = 0.05, with the literal
score alone and with the union rule of union_audit.py (literal and calibrated
paraphrase scores, alpha/2 each). VOUCH verdicts are averaged over K reveal
orders.

Retraining is the exact null: any LEAK or REVOKED there is a false alarm.

Usage: python3 extension/experiments/analyze_baselines.py extension/results/baselines_pythia410m.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from union_audit import cell  # noqa: E402

FEATS = ("loss", "mink", "zlib", "ref")


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), by ranks."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(pos, neg, rng, b=1000):
    vals = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(b)]
    return np.quantile(vals, [0.025, 0.975])


def mia_cell(job):
    subj, seed = job
    rng = np.random.default_rng(seed)
    out = {}
    designs = {"literal/matched": ("in", "ghost"), "literal/unmatched": ("in", "holdout"),
               "paraphrase/matched": ("in_qa", "ghost_qa")}
    for d, (gp, gn) in designs.items():
        P, N = np.array(subj[gp]), np.array(subj[gn])
        for j, f in enumerate(FEATS):
            pos, neg = -P[:, j], -N[:, j]
            a = auc(pos, neg)
            lo, hi = boot_ci(pos, neg, rng)
            out[f"{d}/{f}"] = {"auc": float(a), "lo": float(lo), "hi": float(hi),
                               "leak": bool(lo > 0.5 or hi < 0.5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    runs = json.load(open(args.path))
    keys = [(r["seed"], nm) for r in runs for nm in r["subjects"]]
    with Pool(args.procs) as pool:
        mia = pool.map(mia_cell, [(r["subjects"][nm], r["seed"]) for r in runs
                                  for nm in r["subjects"]])
        vou = pool.map(cell, [(r["subjects"][nm]["pair_diffs"], 0.05, 0.2, args.k)
                              for r in runs for nm in r["subjects"]])
    order = ["none", "retrain"] + sorted({nm for _, nm in keys} - {"none", "retrain"},
                                         key=lambda s: (s.split("@")[0], int(s.split("@")[1])))
    rows = []
    mkeys = list(mia[0].keys())
    print(f"{'subject':9s} {'n':>2s}  VOUCH revoked: lit union | MIA leak fraction "
          f"(literal/matched, literal/unmatched, paraphrase/matched; loss mink zlib ref)")
    for nm in order:
        idx = [i for i, (_, n) in enumerate(keys) if n == nm]
        if not idx:
            continue
        row = {"subject": nm, "n_seeds": len(idx)}
        for rule in ("literal", "union"):
            row[f"vouch_revoked_{rule}"] = float(np.mean([s == "REVOKED" for i in idx
                                                          for s in vou[i][rule]]))
            row[f"vouch_issued_{rule}"] = float(np.mean([s == "ISSUED" for i in idx
                                                         for s in vou[i][rule]]))
        for k in mkeys:
            row[f"mia/{k}/leak"] = float(np.mean([mia[i][k]["leak"] for i in idx]))
            row[f"mia/{k}/auc"] = float(np.mean([mia[i][k]["auc"] for i in idx]))
        rows.append(row)
        g = lambda d: " ".join(f"{row[f'mia/{d}/{f}/leak']:.2f}" for f in FEATS)
        a = lambda d: " ".join(f"{row[f'mia/{d}/{f}/auc']:.2f}" for f in FEATS)
        print(f"{nm:9s} {len(idx):2d}  {row['vouch_revoked_literal']:.2f} {row['vouch_revoked_union']:.2f}"
              f" | {g('literal/matched')} | {g('literal/unmatched')} | {g('paraphrase/matched')}")
        print(f"{'':9s} {'':2s}  {'AUC':>10s} | {a('literal/matched')} | {a('literal/unmatched')} | "
              f"{a('paraphrase/matched')}")
    out = args.out or args.path.replace(".json", "_analysis.json")
    json.dump({"source": os.path.relpath(args.path, ROOT), "eps": 0.2, "alpha": 0.05,
               "k": args.k, "features": list(FEATS), "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
