#!/usr/bin/env python3
"""Reproduce the TOFU/GPT-2 seed-0 pipeline and capture *generations* from the
canary prompts, for the worked example box in the manuscript.

Everything mirrors the benchmark run (same corpus builder, same deterministic
canary manifest, same LoRA/optimiser settings); the only addition is greedy
decoding of the canary questions from the fine-tuned, NPO-unlearned, and
retrained models, together with the per-model secret-span NLL gap D (in-twin
minus ghost) that the certificate actually bets on.

Usage:  python experiments/make_example.py --device cpu --dtype fp32
Output: results/example_generations.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run_lm_big import clone_adapter, fresh_adapter, train_adapter
from run_benchmark import load_benchmark

from vouch.canaries import PGCGenerator
from vouch.training.inject import build_finetune_corpus

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def secret_nll(model, tok, prompt, secret, device):
    """Token-mean NLL of the secret span given the prompt (the benchmark score)."""
    with torch.no_grad():
        p_ids = tok(prompt)["input_ids"]
        s_ids = tok(secret)["input_ids"]
        ids = torch.tensor([p_ids + s_ids], device=device)
        logits = model(input_ids=ids).logits[0, :-1].float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = ids[0, 1:]
        tl = lp[torch.arange(len(tgt)), tgt]
        return float(-tl[-len(s_ids):].mean())


def generate(model, tok, prompt, device, max_new=14):
    with torch.no_grad():
        ids = torch.tensor([tok(prompt)["input_ids"]], device=device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or 0)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=160)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]
    seed = 0
    torch.manual_seed(seed)

    keep, forget, public, util_eval = load_benchmark("tofu", seed)
    manifest = PGCGenerator(seed=seed, domains=("qa",)).generate(m=args.pairs, wave=0)
    corpus, stats = build_finetune_corpus(keep, forget, [manifest], seed=seed)
    forget_texts = list(forget) + manifest.forget_texts()
    print(f"[example] corpus {stats}", flush=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=dtype).to(device)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg, adapter_name="ft",
                           autocast_adapter_dtype=False).to(dtype)

    tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
               block=args.block, device=device, seed=seed,
               ckpt=os.path.join(RESULTS, "ckpt_example_ft.pt"))
    ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
               block=args.block, device=device, seed=seed)

    # the most-memorised stratum makes the cleanest example
    ex_pairs = [p for p in manifest.pairs if p.repetition == 8][:2]

    t0 = time.time()
    model.set_adapter("ft")
    train_adapter(model, tok, corpus, "ft", **tkw)

    keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=seed)
    fresh_adapter(model, "rt", lcfg)
    train_adapter(model, tok, keep_corpus, "rt",
                  **{**tkw, "ckpt": os.path.join(RESULTS, "ckpt_example_rt.pt")})

    clone_adapter(model, "ft", "npo")
    train_adapter(model, tok, forget_texts, "npo", steps=250,
                  retain=keep, npo_ref="ft", **ukw)

    model.eval()
    out = {"seed": seed, "model": "gpt2", "dataset": "tofu",
           "settings": vars(args), "examples": []}
    for p in ex_pairs:
        rec = {"pair_id": p.pair_id, "repetition": p.repetition, "coin": p.coin,
               "in_prompt": p.in_twin[0], "in_secret": p.in_twin[1],
               "ghost_prompt": p.ghost_twin[0], "ghost_secret": p.ghost_twin[1],
               "models": {}}
        for name in ("ft", "npo", "rt"):
            model.set_adapter(name)
            gen = generate(model, tok, p.in_twin[0], device)
            nll_in = secret_nll(model, tok, p.in_twin[0], p.in_twin[1], device)
            nll_gh = secret_nll(model, tok, p.ghost_twin[0], p.ghost_twin[1], device)
            rec["models"][name] = {"generation": gen,
                                   "nll_in": nll_in, "nll_ghost": nll_gh,
                                   "D": nll_in - nll_gh}
            print(f"[{name}] pair {p.pair_id}: gen={gen!r} "
                  f"D={nll_in - nll_gh:+.3f}", flush=True)
        out["examples"].append(rec)
    out["wall_seconds"] = time.time() - t0

    path = os.path.join(RESULTS, "example_generations.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
