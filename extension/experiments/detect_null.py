#!/usr/bin/env python3
"""Null references for the n-gram canary detector (extension, Table 1).

For the organic-matched design, refits the character n-gram detector of
detectability.py with permuted labels (the detector's own null distribution of
out-of-fold AUC), and runs it on two independent organic samples (a same-
distribution control). Also simulates the advantage max|TPR-FPR| of a detector
whose scores are independent of the label, at the sample size of Table 1.

Usage: python3 extension/experiments/detect_null.py [--perms 30]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from detectability import oof_scores, summarize  # noqa: E402
from extension.semantic.canaries import MatchedGenerator  # noqa: E402
from vouch.training.inject import synthetic_bio_corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()
    org = synthetic_bio_corpus(args.n, seed=11)
    can = [p.in_text for p in MatchedGenerator(seed=5).generate(args.n).pairs]
    X = can + org
    y = np.array([1] * args.n + [0] * args.n)
    real = summarize(y, oof_scores(X, y, "ngram"))
    rng = np.random.default_rng(0)
    perm = [summarize(yp, oof_scores(X, yp, "ngram"))["auc"]
            for yp in (rng.permutation(y) for _ in range(args.perms))]
    org2 = synthetic_bio_corpus(args.n, seed=99)
    same = summarize(y, oof_scores(org2 + org, y, "ngram"))
    from sklearn.metrics import roc_curve
    adv = []
    for _ in range(args.sims):
        f, t, _ = roc_curve(y, rng.random(len(y)))
        adv.append(float(np.max(np.abs(t - f))))
    out = {"matched_ngram": real, "perm_auc": perm,
           "perm_auc_mean": float(np.mean(perm)), "perm_auc_sd": float(np.std(perm)),
           "perm_auc_min": float(np.min(perm)), "organic_vs_organic_ngram": same,
           "null_adv_q95": float(np.quantile(adv, 0.95)), "null_adv_q99": float(np.quantile(adv, 0.99))}
    json.dump(out, open(os.path.join(ROOT, "extension", "results", "detect_null.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k != "perm_auc"})


if __name__ == "__main__":
    main()
