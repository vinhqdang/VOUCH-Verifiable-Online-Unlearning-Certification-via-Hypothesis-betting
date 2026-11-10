#!/usr/bin/env python3
"""Literal-only audit versus a per-score union audit (extension, direction 1).

Re-runs the VOUCH verifier offline on the stored per-pair score differences of a
squeezing sweep. For every (seed, subject) cell and K reveal orders it compares

  literal    the literal score alone at level alpha
  union      the literal score and the paraphrase scores (belief_cal,
             belief_max_cal) audited separately at alpha/2 each (Bonferroni);
             REVOKED if either revokes, ISSUED only if both issue
  mixture    all six scores in one engine at level alpha (the class "all" of
             run_semantic.py), whose revocation e-process averages over scores

and reports, per subject, the fraction of (seed, order) runs ending REVOKED and
ISSUED under each rule. Under the exact null each rule's false-alarm rate is at
most alpha (the union by the Bonferroni bound).

Usage: python3 extension/experiments/union_audit.py RESULTS.json [--seeds ...] [--k 20]
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

PARA = ["belief_cal", "belief_max_cal"]
ALL = ["literal", "belief", "belief_max", "literal_cal", "belief_cal", "belief_max_cal"]


def _run(diffs, scores, alpha, eps, seed):
    v = VouchVerifier(scores, VouchConfig(eps=eps, alpha=alpha))
    return v.run([{k: d[k] for k in scores} for d in diffs], shuffle_seed=seed,
                 reveal_source="seeded", early_stop=True).status


def cell(job):
    diffs, alpha, eps, k = job
    out = {"literal": [], "union": [], "mixture": []}
    for j in range(k):
        lit = _run(diffs, ["literal"], alpha, eps, 1000 + j)
        lit2 = _run(diffs, ["literal"], alpha / 2, eps, 1000 + j)
        par2 = _run(diffs, PARA, alpha / 2, eps, 1000 + j)
        mix = _run(diffs, ALL, alpha, eps, 1000 + j)
        uni = ("REVOKED" if "REVOKED" in (lit2, par2) else
               "ISSUED" if lit2 == par2 == "ISSUED" else "UNDETERMINED")
        out["literal"].append(lit)
        out["union"].append(uni)
        out["mixture"].append(mix)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    runs = json.load(open(args.path))
    if args.seeds is not None:
        runs = [r for r in runs if r["seed"] in args.seeds]
    keys = [(r["seed"], nm) for r in runs for nm in r["subjects"]]
    jobs = [(r["subjects"][nm]["pair_diffs"], args.alpha, args.eps, args.k)
            for r in runs for nm in r["subjects"]]
    with Pool(args.procs) as pool:
        res = pool.map(cell, jobs)
    names = sorted({nm for _, nm in keys},
                   key=lambda s: (s != "none", s.split("@")[0],
                                  int(s.split("@")[1]) if "@" in s else 0))
    rows = []
    print(f"{'subject':9s} {'n':>2s}  {'revoked: lit union mix':>24s}  {'issued: lit union mix':>22s}"
          f"  {'lit ISS & union REV':>19s}")
    for nm in names:
        R = [res[i] for i, (_, n) in enumerate(keys) if n == nm]
        f = lambda rule, st: float(np.mean([s == st for r in R for s in r[rule]]))
        caught = float(np.mean([a == "ISSUED" and b == "REVOKED"
                                for r in R for a, b in zip(r["literal"], r["union"])]))
        row = {"subject": nm, "n_seeds": len(R), "k": args.k,
               **{f"revoked_{u}": f(u, "REVOKED") for u in ("literal", "union", "mixture")},
               **{f"issued_{u}": f(u, "ISSUED") for u in ("literal", "union", "mixture")},
               "literal_issued_union_revoked": caught}
        rows.append(row)
        print(f"{nm:9s} {len(R):2d}  {row['revoked_literal']:8.2f} {row['revoked_union']:5.2f} "
              f"{row['revoked_mixture']:4.2f}  {row['issued_literal']:7.2f} {row['issued_union']:5.2f} "
              f"{row['issued_mixture']:4.2f}  {caught:19.2f}", flush=True)
    json.dump({"source": os.path.relpath(args.path, ROOT), "seeds": sorted({s for s, _ in keys}),
               "eps": args.eps, "alpha": args.alpha, "rows": rows},
              open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
