#!/usr/bin/env python3
"""LiRA on a real model: the shadow-model attack of run_lira.py, on GPT-2.

Section 5.13 (`run_lira.py`) runs the shadow-model attack on the TinyGPT tier
because a shadow study repeats the whole pipeline N times, which the
124M-5.1B tiers could not afford on CPU. On a GPU this is affordable, and the
letter's own stated limitation is precisely "one target model, on the TinyGPT
tier" -- so this script runs the same attack, unchanged in design, on the
124M-parameter model the paper actually certifies NPO on.

Reuses the exact TOFU/GPT-2 corpus and canary setup of run_benchmark.py (same
domains, same manifest construction) so the realised advantages here are
directly comparable to the certified NPO row of Table 10 (the same tolerance,
the same architecture, a cohort of the same size).

Each pipeline call (the target, and every shadow) gets its own fresh model:
a new frozen base plus a new LoRA adapter, trained and scored, then
discarded -- exactly the TinyGPT version's per-shadow "fresh TinyGPT()"
pattern, just with an HF model in place of the in-repo one. This costs a
model reload per shadow (a few seconds on a GPU) in exchange for never
having to reason about adapter state carried over between shadows.

Writes the same JSON schema as run_lira.py (target_coins, shadow_coins,
target_phi, shadow_phi), so experiments/analyze_lira.py consumes either
file unchanged -- pass --infile to select which.

Usage:
  python experiments/run_lira_hf.py --shadows 24 --pairs 384 --device cuda
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
sys.path.insert(0, os.path.dirname(__file__))

from run_benchmark import load_benchmark
from run_lm_big import clone_adapter, train_adapter

from vouch.canaries import PGCGenerator
from vouch.training.inject import build_finetune_corpus

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tofu")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seed", type=int, default=0, help="target-model seed")
    ap.add_argument("--shadows", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--npo-steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=160)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--subjects", nargs="+", default=["none", "npo"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="lira_gpt2.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device = args.device
    out_path = os.path.join(RESULTS, args.out)
    state = json.load(open(out_path)) if os.path.exists(out_path) else {}
    state.setdefault("config", {k: v for k, v in vars(args).items()})
    state.setdefault("shadow_phi", {})     # subject -> [shadow][pair][twin]
    state.setdefault("shadow_coins", [])   # [shadow][pair]
    state.setdefault("target_phi", {})     # subject -> [pair][twin]

    keep, forget, public, util_eval = load_benchmark(args.dataset, args.seed)
    domains = ("qa",) if args.dataset == "tofu" else ("pii", "fact")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mcfg = AutoConfig.from_pretrained(args.model)
    if getattr(mcfg, "pad_token_id", None) is None:
        mcfg.pad_token_id = getattr(mcfg, "eos_token_id", 0) or 0
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    def manifest_with_coins(seed_, coins=None):
        man = PGCGenerator(seed=seed_, domains=domains).generate(m=args.pairs, wave=0)
        if coins is not None:
            for p, b in zip(man.pairs, coins):
                p.coin = int(b)
        return man

    target_manifest = manifest_with_coins(args.seed)
    twins = [(p.prefix0, p.secret0, p.prefix1, p.secret1)
             for p in target_manifest.pairs]
    state["target_coins"] = [p.coin for p in target_manifest.pairs]
    state["repetitions"] = [p.repetition for p in target_manifest.pairs]

    tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
               block=args.block, device=device, seed=args.seed)
    ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
               block=args.block, device=device, seed=args.seed)

    def run_pipeline(man, seed_):
        """Fresh model, trained and scored, then discarded -- one full
        pipeline instance per call, matching run_lira.py's TinyGPT pattern."""
        base = AutoModelForCausalLM.from_pretrained(
            args.model, config=mcfg, dtype=torch.float32).to(device)
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(torch.float32)

        @torch.no_grad()
        def logprob(prefix, target):
            model.eval()
            ids = torch.tensor([tok(prefix)["input_ids"] + tok(target)["input_ids"]],
                               device=device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]
            tl = lp[torch.arange(len(tgt)), tgt]
            return float(tl[-len(tok(target)["input_ids"]):].mean())

        def phi_all(adapter):
            model.set_adapter(adapter)
            return [[logprob(p0, s0), logprob(p1, s1)] for p0, s0, p1, s1 in twins]

        corpus, _ = build_finetune_corpus(keep, forget, [man], seed=seed_)
        forget_texts = list(forget) + man.forget_texts()
        res = {}
        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in args.subjects:
            res["none"] = phi_all("ft")
        if "npo" in args.subjects:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget_texts, "npo", steps=args.npo_steps,
                          retain=keep, npo_ref="ft", **ukw)
            res["npo"] = phi_all("npo")
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()
        return res

    if not state["target_phi"]:
        t0 = time.time()
        print("[target] training", flush=True)
        state["target_phi"] = run_pipeline(target_manifest, args.seed)
        json.dump(state, open(out_path, "w"), indent=2, default=float)
        print(f"[target] done in {time.time()-t0:.0f}s", flush=True)

    for rep in range(args.shadows):
        if len(state["shadow_coins"]) > rep:
            continue
        t0 = time.time()
        rng = np.random.default_rng(10_000 + rep)
        coins = [int(b) for b in rng.integers(0, 2, size=args.pairs)]
        man = manifest_with_coins(args.seed, coins)
        print(f"[shadow {rep+1}/{args.shadows}] training", flush=True)
        phis = run_pipeline(man, args.seed)
        for subj, ph in phis.items():
            state["shadow_phi"].setdefault(subj, []).append(ph)
        state["shadow_coins"].append(coins)
        json.dump(state, open(out_path, "w"), indent=2, default=float)
        print(f"[shadow {rep+1}] done in {time.time()-t0:.0f}s", flush=True)

    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
