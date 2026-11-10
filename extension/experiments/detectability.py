#!/usr/bin/env python3
"""Lower bounds on the distinguishing advantage gamma (extension, directions 2 and 4).

Theorem 3 of the theory note bounds the gap between the canary advantage and the
organic advantage by 2*gamma, where gamma is the best advantage with which a
provider can tell canary records from organic records. Data can only bound gamma
from below: every detector we build gives a lower bound. This script builds the
strongest cheap detectors we have and reports, for each canary design,

  AUC     area under the ROC curve (out of fold)
  adv     max_threshold (TPR - FPR), the detector's distinguishing advantage
          and therefore a lower bound on gamma

Canary designs compared against the same organic records (fictitious biographies,
the keep/forget data of the experiments):
  vouch_qa          VOUCH's random-string QA canaries
  semantic_literal  semantic twins, literal statement form
  semantic_qa       semantic twins, question-answer form
  matched           organic-matched twins: drawn from the organic record
                    generator itself, one slot varied (gamma = 0 by design)

Detectors: base-model perplexity features (GPT-2, never fine-tuned), character
n-gram logistic regression, and both together. All out of fold (5-fold).

Usage: python3 extension/experiments/detectability.py [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from extension.semantic.canaries import MatchedGenerator, SemanticGenerator  # noqa: E402
from vouch.canaries import PGCGenerator  # noqa: E402
from vouch.training.inject import synthetic_bio_corpus  # noqa: E402

OUT = os.path.join(ROOT, "extension", "results")


def nll_features(texts, model, tok, device):
    import torch
    feats = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt")["input_ids"].to(device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, -1)[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
            x = -lp.cpu().numpy()
            feats.append([x.mean(), x.max(), np.median(x), x.std(), len(x)])
    return np.array(feats)


def oof_scores(X, y, kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y):
        if kind == "ngram":
            clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                                min_df=2), LogisticRegression(max_iter=2000, C=4))
            clf.fit([X[i] for i in tr], y[tr])
            s[te] = clf.predict_proba([X[i] for i in te])[:, 1]
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def summarize(y, s):
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(auc), "adv": float(np.max(np.abs(tpr - fpr)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="detectability.json")
    ap.add_argument("--designs", nargs="+", default=None,
                    help="subset of designs to run (default: all)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()

    organic = synthetic_bio_corpus(args.n, seed=11)
    vq = PGCGenerator(seed=5, domains=("qa",)).generate(args.n)
    sem = SemanticGenerator(seed=5).generate(args.n)
    designs = {
        "vouch_qa": [p.in_text for p in vq.pairs],
        "semantic_literal": [p.in_text for p in sem.pairs],
        "semantic_qa": [p.qa(p.in_value, 0) for p in sem.pairs],
        "matched": [p.in_text for p in MatchedGenerator(seed=5).generate(args.n).pairs],
    }
    if args.designs:
        designs = {k: v for k, v in designs.items() if k in args.designs}
    org_feats = nll_features(organic, model, tok, args.device)
    report = {"n": args.n, "model": args.model, "designs": {}}
    for name, texts in designs.items():
        y = np.array([1] * len(texts) + [0] * len(organic))
        feats = np.vstack([nll_features(texts, model, tok, args.device), org_feats])
        allt = texts + organic
        sp, sn = oof_scores(feats, y, "dense"), oof_scores(allt, y, "ngram")
        r = {"perplexity": summarize(y, sp), "ngram": summarize(y, sn),
             # out-of-fold scores (high = canary) and labels, for the selective-provider
             # analysis of extension/experiments/selective_provider.py
             "scores": {"y": y.tolist(), "perplexity": sp.tolist(), "ngram": sn.tolist()}}
        r["gamma_lower_bound"] = max(r[k]["adv"] for k in ("perplexity", "ngram"))
        r["examples"] = texts[:3]
        report["designs"][name] = r
        print(f"{name:18s} perplexity AUC={r['perplexity']['auc']:.3f} adv={r['perplexity']['adv']:.3f}"
              f"  ngram AUC={r['ngram']['auc']:.3f} adv={r['ngram']['adv']:.3f}"
              f"  gamma >= {r['gamma_lower_bound']:.3f}", flush=True)
    report["organic_examples"] = organic[:3]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, args.out)
    if os.path.exists(path):          # merge with earlier designs
        old = json.load(open(path))
        if old.get("n") == args.n and old.get("model") == args.model:
            old["designs"].update(report["designs"])
            report = old
    json.dump(report, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
