#!/usr/bin/env python3
"""What a canary detector buys a selective provider (extension, Theorem 5).

A selective provider unlearns the records its detector calls canary-like and leaves the
rest, so records it flags as organic-like keep leaking. Writing B for the set of
organic-like records, a cohort of canaries shows advantage Delta = P_canary(B) and organic
records show Delta^o = P_organic(B). The audit certifies at tolerance eps when
P_canary(B) < eps, and the provider's best choice under a detector with score s (high =
canary) is the threshold test B = {s < t}. Its organic leak is therefore the detector's
true-negative rate at a canary miss rate of eps. Theorem 5 says that the best any
detector can do is the Neyman--Pearson power beta(eps); this script measures what our two
detectors achieve, with the threshold chosen on training folds only (five-fold
cross-fitting) so that the estimate is not tuned on the records it is evaluated on.

Input: extension/results/detectability_scores.json (written by detectability.py).
Usage: python3 extension/experiments/selective_provider.py
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EPS = [0.05, 0.1, 0.2]
NAMES = {"vouch_qa": "VOUCH random-string QA", "semantic_literal": "semantic twin, literal",
         "semantic_qa": "semantic twin, question--answer", "matched": "organic-matched twin"}


def leak(scores, y, eps, folds=5, seed=0):
    """Cross-fitted organic leak and canary miss rate at target miss rate eps."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        ii = idx[y == cls]
        fold[ii] = rng.permutation(len(ii)) % folds
    miss, org = [], []
    for f in range(folds):
        tr, te = fold != f, fold == f
        t = np.quantile(scores[tr & (y == 1)], eps)         # canaries below t are "missed"
        miss.append(np.mean(scores[te & (y == 1)] < t))
        org.append(np.mean(scores[te & (y == 0)] < t))
    return float(np.mean(miss)), float(np.mean(org))


def main():
    d = json.load(open(os.path.join(ROOT, "extension", "results", "detectability_scores.json")))
    out = {"eps": EPS, "designs": {}}
    print(f"{'design':34s} {'detector':11s} " + " ".join(f"eps={e:<4}: miss/leak" for e in EPS))
    for k in NAMES:
        r = d["designs"][k]
        y = np.array(r["scores"]["y"])
        out["designs"][k] = {}
        for det in ("perplexity", "ngram"):
            s = np.array(r["scores"][det])
            res = {str(e): dict(zip(("canary_miss", "organic_leak"), leak(s, y, e))) for e in EPS}
            out["designs"][k][det] = res
            print(f"{NAMES[k]:34s} {det:11s} " + " ".join(
                f"{res[str(e)]['canary_miss']:.2f}/{res[str(e)]['organic_leak']:.2f}      " for e in EPS))
    json.dump(out, open(os.path.join(ROOT, "extension", "results", "selective_provider.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
