#!/usr/bin/env python3
"""VOUCH end-to-end for multi-billion-parameter HF models on a small GPU.

Memory design (fits 5B models on a 16 GB T4):
  * ONE frozen fp16/bf16 base model instance, shared by every stage;
  * every training stage (fine-tune, retrain, unlearning, relearn probe)
    is a LoRA adapter on that base; adapters are cloned/switched in place,
    never the base;
  * NPO's frozen reference = the fine-tuned adapter itself: nll_ref is a
    no-grad forward with the "ft" adapter active, nll_theta a grad forward
    with the trainable clone -- no second model in memory;
  * quantization probe (P2) is skipped (would materialize merged weights);
    P1 relearn and P3 jailbreak run as usual.

"Retrain" = fresh adapter on the pristine base, trained on keep-only data:
exact unlearning within the adapter-FT paradigm (the base never saw any
canary).

Usage (Gemma 4, 2026):
  python experiments/run_lm_big.py --model google/gemma-4-E2B-it \\
      --seeds 0 --pairs 512 --eps 0.2 --dtype fp16 --tag gemma4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.canaries import PGCGenerator
from vouch.probes.probes import JAILBREAK_WRAPPERS
from vouch.training.inject import build_finetune_corpus, synthetic_bio_corpus
from vouch.verify import ScoreEngine, VouchConfig, VouchVerifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------
# training primitives on the shared-base / multi-adapter model
# --------------------------------------------------------------------------

def batches(texts, bs, rng):
    idx = list(range(len(texts)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            yield [texts[j] for j in idx[i:i + bs]]


def encode_batch(tok, texts, block, device):
    ids = [tok(t, truncation=True, max_length=block)["input_ids"] for t in texts]
    mx = max(len(s) for s in ids)
    pad = tok.pad_token_id or 0
    out = torch.full((len(ids), mx), pad, dtype=torch.long)
    for r, s in enumerate(ids):
        out[r, : len(s)] = torch.tensor(s)
    return out.to(device)


def seq_nll(model, batch, pad_id):
    logits = model(input_ids=batch).logits.float()
    tgt = batch[:, 1:]
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != pad_id).float()
    return -(tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)


def train_adapter(model, tok, texts, adapter, steps, bs, lr, block, device,
                  seed=0, sign=+1, retain=None, retain_w=1.0, npo_ref=None,
                  beta=0.1, log_every=100, ckpt=None, ckpt_every=100):
    """Train `adapter` (already active) on texts.

    sign=+1: descent (fine-tune);  sign=-1: ascent (GA / GradDiff forget term).
    npo_ref: name of a frozen reference adapter -> NPO loss instead of CE.
    ckpt: optional path for intra-stage checkpointing (adapter + optimizer +
    step) so short-lived VMs make progress through long stages.
    """
    from peft.utils import (get_peft_model_state_dict,
                            set_peft_model_state_dict)
    rng = random.Random(("train", adapter, seed).__repr__())
    pad = tok.pad_token_id or 0
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    start_step = 0
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        set_peft_model_state_dict(model, state["adapter"], adapter_name=adapter)
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"    [{adapter}] resumed at step {start_step}", flush=True)
    gen = batches(texts, bs, rng)
    rgen = batches(retain, bs, rng) if retain else None
    model.train()
    for step in range(steps):
        if step < start_step:
            next(gen)                      # keep batch order aligned
            if rgen is not None:
                next(rgen)
            continue
        b = encode_batch(tok, next(gen), block, device)
        if npo_ref is not None:
            with torch.no_grad():
                model.set_adapter(npo_ref)
                nll_ref = seq_nll(model, b, pad)
            model.set_adapter(adapter)
            nll_theta = seq_nll(model, b, pad)
            loss = (2.0 / beta) * F.softplus(beta * (nll_ref - nll_theta)).mean()
        else:
            loss = sign * seq_nll(model, b, pad).mean()
        if rgen is not None:
            rb = encode_batch(tok, next(rgen), block, device)
            loss = loss + retain_w * seq_nll(model, rb, pad).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"    [{adapter}] step {step+1}/{steps} loss {loss.item():.4f}",
                  flush=True)
        if ckpt and (step + 1) % ckpt_every == 0 and (step + 1) < steps:
            torch.save({"adapter": get_peft_model_state_dict(
                            model, adapter_name=adapter),
                        "opt": opt.state_dict(), "step": step + 1}, ckpt)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)                    # stage finished; final ckpt is
                                           # saved by the caller


def clone_adapter(model, src, dst):
    """Register adapter `dst` as a copy of `src` and make it active/trainable."""
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    sd = copy.deepcopy(get_peft_model_state_dict(model, adapter_name=src))
    model.add_adapter(dst, model.peft_config[src])
    set_peft_model_state_dict(model, sd, adapter_name=dst)
    model.set_adapter(dst)
    for n, p in model.named_parameters():
        p.requires_grad = (f".{dst}." in n)
    return model


def fresh_adapter(model, name, cfg):
    model.add_adapter(name, cfg)
    model.set_adapter(name)
    for n, p in model.named_parameters():
        p.requires_grad = (f".{name}." in n)
    return model


def drop_adapter(model, name):
    model.delete_adapter(name)
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=512)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--corpus-scale", type=float, default=1.0)
    ap.add_argument("--methods", nargs="+",
                    default=["none", "retrain", "ga", "grad_diff", "npo",
                             "npo_P1_relearn", "npo_P3_jailbreak"])
    ap.add_argument("--tag", default="big")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    all_out = []
    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype).to(device)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                          lora_dropout=0.0, target_modules="all-linear",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg, adapter_name="ft").to(dtype)

        # Phase 0
        keep = synthetic_bio_corpus(int(3000 * args.corpus_scale), seed=seed)
        forget = synthetic_bio_corpus(int(500 * args.corpus_scale), seed=seed + 777)
        manifest = PGCGenerator(seed=seed).generate(m=args.pairs, wave=0)
        commitment = manifest.commitment()
        corpus, stats = build_finetune_corpus(keep, forget, [manifest], seed=seed)
        forget_texts = list(forget) + manifest.forget_texts()
        print(f"[seed {seed}] {args.model} corpus {stats}", flush=True)

        def logprob_fn():
            @torch.no_grad()
            def fn(prefix, target):
                model.eval()
                p_ids = tok(prefix)["input_ids"]
                t_ids = tok(target)["input_ids"]
                ids = torch.tensor([p_ids + t_ids], device=device)
                logits = model(input_ids=ids).logits[0, :-1].float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = ids[0, 1:]
                tl = lp[torch.arange(len(tgt)), tgt]
                return tl[-len(t_ids):].cpu().numpy()
            return fn

        results = {"seed": seed, "backend": args.model, "m_pairs": args.pairs,
                   "eps": args.eps, "alpha": args.alpha, "lora_r": args.lora_r,
                   "dtype": args.dtype, "manifest_sha256": commitment,
                   "corpus_stats": stats, "certs": {}}

        def verify(tag, adapter, wrappers=None):
            t_v = time.time()
            model.set_adapter(adapter)
            eng = ScoreEngine(logprob_fn(), n_queries=args.queries)
            if wrappers is not None:
                eng.wrappers = wrappers
            diffs = [eng.pair_differences(p.in_twin, p.ghost_twin)
                     for p in manifest.pairs]
            v = VouchVerifier(eng.score_names,
                              VouchConfig(eps=args.eps, alpha=args.alpha),
                              manifest_sha256=commitment)
            cert = v.run(diffs, shuffle_seed=seed, early_stop=True)
            md = float(np.mean([d["loss"] for d in diffs]))
            print(f"[seed {seed}] {tag:12s} status={cert.status:12s} "
                  f"t={cert.t_stop:4d} logEcert={cert.log_e_cert:7.2f} "
                  f"logErev={cert.log_e_rev:7.2f} dU={cert.delta_upper:6.3f} "
                  f"meanD={md:6.3f}", flush=True)
            rec = json.loads(cert.to_json())
            rec.update(mean_loss_diff=md, pair_diffs=diffs,
                       scoring_seconds=time.time() - t_v)
            results["certs"][tag] = rec
            # partial save after every verification: sessions can be
            # reclaimed mid-run, results must survive
            with open(os.path.join(RESULTS, f"lm_e2e_{args.tag}.json"), "w") as f:
                json.dump(all_out + [results], f, indent=2, default=float)

        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=device, seed=seed)

        M = set(args.methods)
        # Phase 0': fine-tune adapter on corpus (includes in-twins)
        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in M:
            verify("none", "ft")

        if "retrain" in M:
            keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=seed)
            fresh_adapter(model, "rt", lcfg)
            train_adapter(model, tok, keep_corpus, "rt", **tkw)
            verify("retrain", "rt")
            drop_adapter(model, "rt")

        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=device, seed=seed)

        if "ga" in M:
            clone_adapter(model, "ft", "ga")
            train_adapter(model, tok, forget_texts, "ga", steps=100, sign=-1, **ukw)
            verify("ga", "ga")
            drop_adapter(model, "ga")

        if "grad_diff" in M:
            clone_adapter(model, "ft", "gd")
            train_adapter(model, tok, forget_texts, "gd", steps=250, sign=-1,
                          retain=keep, **ukw)
            verify("grad_diff", "gd")
            drop_adapter(model, "gd")

        if "npo" in M:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget_texts, "npo", steps=250,
                          retain=keep, npo_ref="ft", **ukw)
            verify("npo", "npo")
            if "npo_P1_relearn" in M:
                clone_adapter(model, "npo", "p1")
                public = synthetic_bio_corpus(400, seed=seed + 4242)
                train_adapter(model, tok, public, "p1", steps=100, **ukw)
                verify("npo_P1_relearn", "p1")
                drop_adapter(model, "p1")
            if "npo_P3_jailbreak" in M:
                verify("npo_P3_jailbreak", "npo", wrappers=JAILBREAK_WRAPPERS)
            drop_adapter(model, "npo")

        results["wall_seconds"] = time.time() - t0
        all_out.append(results)
        path = os.path.join(RESULTS, f"lm_e2e_{args.tag}.json")
        with open(path, "w") as f:
            json.dump(all_out, f, indent=2, default=float)
        print(f"[saved] {path}", flush=True)
        del model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
