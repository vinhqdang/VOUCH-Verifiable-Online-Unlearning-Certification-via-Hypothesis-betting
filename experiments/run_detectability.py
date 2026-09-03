#!/usr/bin/env python3
"""Can a provider spot the planted canaries?

The honest-but-verifiable threat model assumes the provider does not
special-case the committed cohort.  That assumption is doing real work, so
this experiment measures how hard the cohort would be to find if a provider
tried, using the two cheapest and most obvious filters:

  P1  a perplexity filter -- score every corpus record under the *pristine
      base model* (which has never seen any canary) and threshold;
  P2  a trained detector -- a bag-of-character-n-gram logistic regression
      fitted on a labelled sample of canaries against organic records and
      evaluated out-of-fold.

Both are run on the exact corpora the end-to-end experiments build, at the
canary share those experiments use, and separately at the far smaller
deployment share.  We report AUC, average precision, and -- the operationally
meaningful quantity -- the fraction of the cohort a provider recovers at a
1% false-positive rate on organic data, together with how much organic data
they would have to discard to get it.

A high detectability number is a finding about the *template library*, not
about the framework: the audit's soundness rests on the coin, not on
concealment, and the paper reports the number either way.

Usage:
  python3 experiments/run_detectability.py --dataset tofu --model gpt2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from vouch.canaries import PGCGenerator                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def tpr_at_fpr(y, s, fpr_target=0.01):
    """Recall of the positive class at a fixed false-positive rate."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    neg = np.sort(s[~y])
    if len(neg) == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1.0 - fpr_target))
    return float((s[y] > thr).mean()), thr


def auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tofu", "muse"], default="tofu")
    ap.add_argument("--model", default="gpt2",
                    help="pristine base model used for the perplexity filter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--block", type=int, default=160)
    ap.add_argument("--max-organic", type=int, default=3000)
    ap.add_argument("--domains", default="",
                    help="comma-separated template names; default is the "
                         "benchmark's own (qa for TOFU, pii+fact for MUSE)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from run_benchmark import load_benchmark
    from run_lm_big import encode_batch, seq_nll

    t0 = time.time()
    keep, forget, public, util_eval = load_benchmark(args.dataset, args.seed)
    organic = (keep + forget)[: args.max_organic]
    default_dom = ("qa",) if args.dataset == "tofu" else ("pii", "fact")
    if args.domains:
        variants = [("custom",
                     tuple(x.strip() for x in args.domains.split(",") if x.strip()))]
    else:
        # the deployed template, its word-composed variant, and the
        # entropy-diluted variant, so the design rule can be read off
        variants = [("deployed", default_dom)]
        if args.dataset == "tofu":
            variants += [("word_composed", ("qa_nat",)),
                         ("entropy_diluted", ("qa_diluted",))]
        else:
            variants += [("word_composed", ("fact_nat",)),
                         ("entropy_diluted", ("fact_diluted",))]
    cohorts = {}
    for nm, dom in variants:
        man = PGCGenerator(seed=args.seed, domains=dom).generate(
            m=args.pairs, wave=0)
        # a provider can only look at what is *in* the corpus: the in-twins
        cohorts[nm] = ([p.in_text for p in man.pairs], list(dom))
    print(f"[data] {len(organic)} organic records; variants "
          f"{list(cohorts)} x {args.pairs} canaries", flush=True)

    out = {"dataset": args.dataset, "base_model": args.model,
           "seed": args.seed, "n_pairs": args.pairs,
           "n_organic": len(organic), "variants": {}}

    # ---- pristine base model for the perplexity filter -------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(args.device).eval()
    pad = tok.pad_token_id or 0

    @torch.no_grad()
    def nlls(texts, bs=16):
        vals = []
        for i in range(0, len(texts), bs):
            b = encode_batch(tok, texts[i:i + bs], args.block, args.device)
            vals.extend(seq_nll(base, b, pad).tolist())
        return np.array(vals, dtype=float)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from vouch.canaries.generator import (DILUTED_SECRET_BITS,
                                          NATURAL_SECRET_BITS)

    BITS = {"deployed": 10 * math.log2(36),
            "word_composed": NATURAL_SECRET_BITS,
            "entropy_diluted": DILUTED_SECRET_BITS}

    nll_org = nlls(organic)
    print(f"[organic] mean NLL {nll_org.mean():.3f} +- {nll_org.std():.3f}",
          flush=True)

    for nm, (canaries, dom) in cohorts.items():
        nll_can = nlls(canaries)
        n_tok = float(np.mean([len(tok(t)["input_ids"]) for t in canaries]))
        y = np.r_[np.zeros(len(nll_org)), np.ones(len(nll_can))]
        sc = np.r_[nll_org, nll_can]        # canaries are high-perplexity
        auc, apr = auc_ap(y, sc)
        tpr, thr = tpr_at_fpr(y, sc, 0.01)

        # trained character-n-gram detector, out-of-fold
        texts = list(organic) + list(canaries)
        ylab = np.r_[np.zeros(len(organic)), np.ones(len(canaries))].astype(int)
        scores = np.zeros(len(texts))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        for tr, te in skf.split(texts, ylab):
            clf = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                min_df=2, max_features=50000, sublinear_tf=True),
                LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced"))
            clf.fit([texts[i] for i in tr], ylab[tr])
            scores[te] = clf.predict_proba([texts[i] for i in te])[:, 1]
        auc2, apr2 = auc_ap(ylab, scores)
        tpr2, _ = tpr_at_fpr(ylab, scores, 0.01)

        bits = BITS.get(nm, float("nan"))
        out["variants"][nm] = {
            "domains": dom, "secret_bits": bits, "mean_tokens": n_tok,
            "predicted_excess_nll_per_token": bits * math.log(2) / n_tok,
            "observed_excess_nll_per_token": float(nll_can.mean() - nll_org.mean()),
            "mean_nll_canary": float(nll_can.mean()),
            "sd_nll_canary": float(nll_can.std()),
            "perplexity_filter": {"auc": auc, "average_precision": apr,
                                  "recall_at_1pct_fpr": tpr,
                                  "threshold_nll": thr},
            "trained_detector": {"auc": auc2, "average_precision": apr2,
                                 "recall_at_1pct_fpr": tpr2,
                                 "model": "char_wb 2-4gram tf-idf + logistic "
                                          "regression, 5-fold OOF"},
        }
        print(f"[{nm:16s}] bits {bits:5.1f} tokens {n_tok:5.1f} "
              f"excess/token pred {bits*math.log(2)/n_tok:.3f} "
              f"obs {nll_can.mean()-nll_org.mean():.3f} | "
              f"ppl AUC {auc:.3f} rec@1%FPR {tpr:.3f} | "
              f"clf AUC {auc2:.3f} rec@1%FPR {tpr2:.3f}", flush=True)

    out["mean_nll_organic"] = float(nll_org.mean())
    out["sd_nll_organic"] = float(nll_org.std())
    out["wall_seconds"] = time.time() - t0
    tag = args.tag or f"{args.dataset}_{args.model.split('/')[-1]}"
    path = os.path.join(RESULTS, f"detectability_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
