#!/usr/bin/env python3
"""A LiRA-style shadow-model attack from outside the declared class F.

Reviewer 2's W4 asked for an attack strictly stronger than anything in F,
run against an already-certified model.  The two attacks in Section 5.11
(stratum-restricted, cross-fitted learned combination) came back below their
own binomial nulls, and both are weak by construction: one is a subpopulation
restriction, the other a linear recombination of scores already in F.  The
attack a sceptic actually wants is LiRA \\citep{carlini2022lira}: per-example
calibration of the target model's confidence against the distribution of
confidences that *shadow* models trained with and without that example
produce.

Why LiRA is outside F.  Every score in F is a function of the deployed
model's outputs alone.  LiRA additionally consumes N independently trained
shadow models, so it is not a member of F at any Q, and no reweighting of
scores in F reproduces it.  It is also not a score a *verifier* could run
under VOUCH's own cost model -- it needs N extra training runs -- which is
exactly why it is the right stress test of the class restriction rather
than a candidate for inclusion in it.

Design (online LiRA with matched shadows).  Each shadow repeats the whole
pipeline on the same twin pool with an independent redraw of the inclusion
coins, so across N shadows every twin text is IN in about half of them and
OUT in the other half, giving both populations per twin from one set of
runs.  For each subject we train the matched shadow stage, so a certified
NPO target is calibrated against shadows that were themselves NPO-unlearned
rather than against a pre-unlearning proxy.

For twin x, phi(M, x) is the token-normalised log-likelihood of its secret.
We fit N(mu_in(x), sigma^2) and N(mu_out(x), sigma^2) with a variance pooled
across twins (Carlini et al.'s global-variance variant, which is what modest
shadow counts support) and score

    Lambda(x) = log N(phi_target(x); mu_in(x), sigma^2)
              - log N(phi_target(x); mu_out(x), sigma^2),

the per-twin log-likelihood ratio of "was trained on" against "was not".
The attack's per-pair sign is 1{Lambda(in-twin) > Lambda(ghost-twin)} and its
realised cohort advantage is 2*mean(sign) - 1, directly comparable to the
tolerance and to the in-class advantages of Table 16.

Controls.  The un-unlearned model is the positive control: unless LiRA finds
a large advantage there, it is a weak attack and a null result on a
certified model would mean nothing.  Retraining from a fresh model is the
negative control, where nothing is there to find.

Tier.  This runs on the in-repo TinyGPT tier -- the one the paper already
uses where "retraining is cheap enough to serve as ground truth"
(Section 5.7).  A shadow study needs the whole pipeline repeated N times,
which the 124M--5.1B tiers cannot afford on the compute this study had; the
shadow count is reported with every number it produces.

Usage:
  python experiments/run_lira.py --shadows 24 --pairs 256 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vouch.unlearn.methods as U
from vouch.canaries import PGCGenerator
from vouch.models.tiny_gpt import (CharTokenizer, TinyGPT, TinyGPTConfig,
                                   lm_logprob_fn)
from vouch.training.inject import build_finetune_corpus, synthetic_bio_corpus

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="target-model seed")
    ap.add_argument("--shadows", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--npo-steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--keep-docs", type=int, default=600)
    ap.add_argument("--forget-docs", type=int, default=120)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--subjects", nargs="+", default=["none", "npo", "retrain"])
    ap.add_argument("--out", default="lira_tiny.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_path = os.path.join(RESULTS, args.out)
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}
    state.setdefault("config", vars(args))
    state.setdefault("shadow_phi", {})     # subject -> [shadow][pair][twin]
    state.setdefault("shadow_coins", [])   # [shadow][pair]
    state.setdefault("target_phi", {})     # subject -> [pair][twin]

    tok = CharTokenizer()
    cfg = TinyGPTConfig(vocab_size=tok.vocab_size)
    keep = synthetic_bio_corpus(args.keep_docs, seed=args.seed)
    forget = synthetic_bio_corpus(args.forget_docs, seed=args.seed + 777)

    def manifest_with_coins(coins=None):
        man = PGCGenerator(seed=args.seed).generate(m=args.pairs, wave=0)
        if coins is not None:
            for p, b in zip(man.pairs, coins):
                p.coin = int(b)
        return man

    target_manifest = manifest_with_coins()
    twins = [(p.prefix0, p.secret0, p.prefix1, p.secret1)
             for p in target_manifest.pairs]
    state["target_coins"] = [p.coin for p in target_manifest.pairs]
    state["repetitions"] = [p.repetition for p in target_manifest.pairs]

    def phi_all(model):
        """Token-normalised log-likelihood of every twin's secret, in the
        fixed (twin0, twin1) order so IN/OUT populations line up per twin."""
        fn = lm_logprob_fn(model, tok, "cpu")
        return [[float(np.mean(fn(p0, s0))), float(np.mean(fn(p1, s1)))]
                for p0, s0, p1, s1 in twins]

    def run_pipeline(man, tag):
        corpus, _ = build_finetune_corpus(keep, forget, [man], seed=args.seed)
        forget_texts = list(forget) + man.forget_texts()
        res = {}
        model = TinyGPT(cfg)
        U.finetune(model, tok, corpus, steps=args.steps, batch_size=args.batch,
                   lr=args.lr, seed=args.seed, device="cpu")
        if "none" in args.subjects:
            res["none"] = phi_all(model)
        if "npo" in args.subjects:
            import copy
            m_npo = copy.deepcopy(model)
            U.npo(m_npo, tok, forget_texts, retain_texts=keep,
                  steps=args.npo_steps, batch_size=max(args.batch // 2, 1),
                  lr=args.lr / 2, seed=args.seed, device="cpu")
            res["npo"] = phi_all(m_npo)
            del m_npo
        if "retrain" in args.subjects:
            keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=args.seed)
            m_rt = TinyGPT(cfg)
            U.finetune(m_rt, tok, keep_corpus, steps=args.steps,
                       batch_size=args.batch, lr=args.lr, seed=args.seed,
                       device="cpu")
            res["retrain"] = phi_all(m_rt)
            del m_rt
        del model
        return res

    if not state["target_phi"]:
        t0 = time.time()
        print("[target] training", flush=True)
        state["target_phi"] = run_pipeline(target_manifest, "target")
        json.dump(state, open(out_path, "w"), indent=2, default=float)
        print(f"[target] done in {time.time()-t0:.0f}s", flush=True)

    for rep in range(args.shadows):
        if len(state["shadow_coins"]) > rep:
            continue
        t0 = time.time()
        rng = np.random.default_rng(10_000 + rep)
        coins = [int(b) for b in rng.integers(0, 2, size=args.pairs)]
        man = manifest_with_coins(coins)
        print(f"[shadow {rep+1}/{args.shadows}] training", flush=True)
        phis = run_pipeline(man, f"shadow{rep}")
        for subj, ph in phis.items():
            state["shadow_phi"].setdefault(subj, []).append(ph)
        state["shadow_coins"].append(coins)
        json.dump(state, open(out_path, "w"), indent=2, default=float)
        print(f"[shadow {rep+1}] done in {time.time()-t0:.0f}s", flush=True)

    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
