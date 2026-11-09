#!/usr/bin/env python3
"""Summarise the squeezing sweep (extension, direction 1).

For each unlearning method and strength, over seeds:
  literal / para      mean advantage of the literal score and of the calibrated
                      paraphrase score (belief_cal), with standard errors
  gap                 mean of para - literal (paired by seed)
  squeeze             number of seeds where the literal-only audit ISSUES a
                      certificate while the calibrated paraphrase audit REVOKES:
                      the literal certificate is then misleading
  lit_iss / para_rev  counts of the two verdicts separately

Usage: python3 extension/experiments/summarize_squeeze.py [results.json] [--eps 0.2]
"""
import argparse
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=os.path.join(
        ROOT, "extension", "results", "semantic_pythia410m_squeeze.json"))
    ap.add_argument("--eps", default="0.2")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "extension", "results", "squeeze_summary.json"))
    args = ap.parse_args()
    runs = json.load(open(args.path))
    names = sorted({k for r in runs for k in r["subjects"]},
                   key=lambda s: (s.split("@")[0] != "none", s.split("@")[0],
                                  int(s.split("@")[1]) if "@" in s else 0))
    rows = []
    print(f"{'subject':9s} {'n':>2s} {'literal':>13s} {'para_cal':>13s} {'gap':>13s}"
          f" {'lit_iss':>7s} {'para_rev':>8s} {'squeeze':>7s}")
    for nm in names:
        S = [r["subjects"][nm] for r in runs if nm in r["subjects"]]
        lit = np.array([s["advantage"]["literal"] for s in S])
        par = np.array([s["advantage"]["belief_cal"] for s in S])
        v_l = [s["verdicts"][f"literal/eps={args.eps}"]["status"] for s in S]
        v_p = [s["verdicts"][f"belief_cal/eps={args.eps}"]["status"] for s in S]
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        row = {"subject": nm, "n": len(S),
               "literal": float(lit.mean()), "literal_se": float(se(lit)),
               "para_cal": float(par.mean()), "para_cal_se": float(se(par)),
               "gap": float((par - lit).mean()), "gap_se": float(se(par - lit)),
               "literal_issued": sum(v == "ISSUED" for v in v_l),
               "para_revoked": sum(v == "REVOKED" for v in v_p),
               "squeeze": sum(a == "ISSUED" and b == "REVOKED" for a, b in zip(v_l, v_p))}
        rows.append(row)
        f = lambda m, s: f"{m:+.3f}±{s:.3f}"
        print(f"{nm:9s} {len(S):2d} {f(row['literal'], row['literal_se']):>13s} "
              f"{f(row['para_cal'], row['para_cal_se']):>13s} {f(row['gap'], row['gap_se']):>13s} "
              f"{row['literal_issued']:7d} {row['para_revoked']:8d} {row['squeeze']:7d}")
    json.dump({"source": os.path.relpath(args.path, ROOT), "eps": float(args.eps),
               "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
