#!/usr/bin/env python3
"""What a canary detector buys a selective provider (extension, sharp transfer theorem).

A selective provider unlearns the records its detector calls canary-like and leaves the
rest, so records it flags as organic-like keep leaking. Writing B for the set of
organic-like records, a cohort of canaries shows advantage Delta = P_canary(B) and organic
records show Delta^o = P_organic(B). The audit certifies at tolerance eps when
P_canary(B) < eps, and the provider's best choice under a detector with score s (high =
canary) is the threshold test B = {s < t}. Its organic leak is therefore the detector's
true-negative rate at a canary miss rate of eps. The sharp transfer theorem says that the best any
detector can do is the Neyman--Pearson power beta(eps); this script measures what our two
detectors achieve, with the threshold and the orientation of the score (a provider may
use s or -s) chosen on training folds only (five-fold cross-fitting), so that the estimate
is not tuned on the records it is evaluated on. The provider is assumed to hold labelled
canaries, as the detectors are trained on them. The detector scores are themselves
out-of-fold from detectability.py; its fold split differs from the one used here.

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
    """Cross-fitted organic leak and canary miss rate at target miss rate eps.

    On each training split the provider picks the orientation (s or -s) that leaves
    more organic records un-unlearned at canary miss rate eps, and the threshold t as the
    eps-quantile of the canary scores; B = {orientation * s < t} is evaluated on the
    held-out fold."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        ii = idx[y == cls]
        fold[ii] = rng.permutation(len(ii)) % folds
    miss, org = [], []
    for f in range(folds):
        tr, te = fold != f, fold == f
        best = None
        for sign in (1.0, -1.0):
            z = sign * scores
            t = np.quantile(z[tr & (y == 1)], eps)
            train_leak = np.mean(z[tr & (y == 0)] < t)
            if best is None or train_leak > best[0]:
                best = (train_leak, sign, t)
        _, sign, t = best
        z = sign * scores
        miss.append(np.mean(z[te & (y == 1)] < t))
        org.append(np.mean(z[te & (y == 0)] < t))
    return float(np.mean(miss)), float(np.mean(org))


def main():
    d = json.load(open(os.path.join(ROOT, "extension", "results", "detectability_scores.json")))
    from collections import Counter
    import sys
    sys.path.insert(0, ROOT)
    from vouch.training.inject import synthetic_bio_corpus
    docs = synthetic_bio_corpus(2000, seed=11)
    cnt = Counter(docs)
    out = {"eps": EPS, "designs": {},
           "organic_duplicate_share": sum(v for v in cnt.values() if v > 1) / len(docs),
           "organic_duplicate_share_n": len(docs)}
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
