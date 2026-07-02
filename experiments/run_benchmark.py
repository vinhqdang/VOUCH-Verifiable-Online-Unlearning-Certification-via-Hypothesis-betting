#!/usr/bin/env python3
"""VOUCH on the standard unlearning benchmarks: TOFU and MUSE-News.

TOFU  (locuslab/TOFU): fictitious-author QA. keep = retain90 (train part),
      forget = forget10 (400 QA); canaries use the QA template domain so the
      twins match the corpus format. Utility = held-out retain QA NLL.
MUSE  (muse-bench/MUSE-News, raw): BBC news. keep = retain1 chunks,
      forget = forget chunks; canaries use record-style templates.
      Utility = holdout-chunk NLL.

Model default: microsoft/phi-1_5 (the TOFU-standard small model), LoRA r=16
on a frozen fp16 base via the shared-base multi-adapter machinery of
run_lm_big.py (fits a 16 GB T4; sessions can die -> partial saves after
every verification and method-level resume via --resume).

Usage:
  python experiments/run_benchmark.py --dataset tofu --seeds 0 1 2
  python experiments/run_benchmark.py --dataset muse --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run_lm_big import (batches, clone_adapter, drop_adapter, encode_batch,
                        fresh_adapter, seq_nll, train_adapter)

from vouch.canaries import PGCGenerator
from vouch.probes.probes import JAILBREAK_WRAPPERS
from vouch.training.inject import build_finetune_corpus
from vouch.verify import ScoreEngine, VouchConfig, VouchVerifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def load_benchmark(name: str, seed: int):
    """Returns (keep, forget, public, utility_eval) text lists."""
    from datasets import load_dataset
    rng = random.Random(("bench", name, seed).__repr__())
    if name == "tofu":
        fmt = lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}"
        retain = [fmt(r) for r in load_dataset("locuslab/TOFU", "retain90")["train"]]
        forget = [fmt(r) for r in load_dataset("locuslab/TOFU", "forget10")["train"]]
        rng.shuffle(retain)
        keep, util_eval = retain[:-200], retain[-200:]
        wf = [fmt(r) for r in load_dataset("locuslab/TOFU", "world_facts")["train"]]
        return keep, forget, wf, util_eval
    if name == "muse":
        def chunks(split, cap):
            out = []
            for r in split:
                t = r["text"].strip()
                for i in range(0, len(t), 500):
                    c = t[i:i + 500].strip()
                    if len(c) > 100:
                        out.append(c)
                    if len(out) >= cap:
                        return out
            return out
        ds = load_dataset("muse-bench/MUSE-News", "raw")
        keep = chunks(ds["retain1"], 3000)
        forget = chunks(ds["forget"], 500)
        holdout = chunks(ds["holdout"], 600)
        return keep, forget, holdout[:400], holdout[400:600]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tofu", "muse"], required=True)
    ap.add_argument("--model", default="microsoft/phi-1_5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--block", type=int, default=160)
    ap.add_argument("--methods", nargs="+",
                    default=["none", "retrain", "ga", "grad_diff", "npo",
                             "npo_P1_relearn", "npo_P3_jailbreak"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true",
                    help="skip (seed, method) pairs already in the output JSON")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or f"{args.dataset}_{args.model.split('/')[-1]}"
    out_path = os.path.join(RESULTS, f"lm_e2e_{tag}.json")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    all_out = []
    if args.resume and os.path.exists(out_path):
        all_out = json.load(open(out_path))
        print(f"[resume] loaded {len(all_out)} prior seed record(s)", flush=True)

    for seed in args.seeds:
        prior = next((r for r in all_out if r["seed"] == seed), None)
        done = set(prior["certs"].keys()) if prior else set()
        todo = [m for m in args.methods if m not in done]
        if not todo:
            print(f"[seed {seed}] all methods done, skipping", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(seed)

        keep, forget, public, util_eval = load_benchmark(args.dataset, seed)
        # canary domains matched to corpus format (Section 6.7 / D6)
        domains = ("qa",) if args.dataset == "tofu" else ("pii", "fact")
        manifest = PGCGenerator(seed=seed, domains=domains).generate(
            m=args.pairs, wave=0)
        commitment = manifest.commitment()
        corpus, stats = build_finetune_corpus(keep, forget, [manifest], seed=seed)
        forget_texts = list(forget) + manifest.forget_texts()
        print(f"[seed {seed}] {args.dataset}/{args.model} corpus {stats} "
              f"todo={todo}", flush=True)

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype).to(device)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                          lora_dropout=0.0, target_modules="all-linear",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg, adapter_name="ft").to(dtype)
        pad = tok.pad_token_id or 0

        results = prior or {
            "seed": seed, "dataset": args.dataset, "backend": args.model,
            "m_pairs": args.pairs, "eps": args.eps, "alpha": args.alpha,
            "lora_r": args.lora_r, "dtype": args.dtype,
            "manifest_sha256": commitment, "corpus_stats": stats, "certs": {}}
        if prior is None:
            all_out.append(results)

        @torch.no_grad()
        def utility_nll():
            model.eval()
            losses = []
            for i in range(0, len(util_eval), 16):
                b = encode_batch(tok, util_eval[i:i + 16], args.block, device)
                losses.extend(seq_nll(model, b, pad).tolist())
            return float(np.mean(losses))

        def logprob_fn():
            @torch.no_grad()
            def fn(prefix, target):
                model.eval()
                ids = torch.tensor(
                    [tok(prefix)["input_ids"] + tok(target)["input_ids"]],
                    device=device)
                logits = model(input_ids=ids).logits[0, :-1].float()
                lp = torch.log_softmax(logits, dim=-1)
                tgt = ids[0, 1:]
                tl = lp[torch.arange(len(tgt)), tgt]
                n_t = len(tok(target)["input_ids"])
                return tl[-n_t:].cpu().numpy()
            return fn

        def verify(m_tag, adapter, wrappers=None):
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
            md_all = [d["loss"] for d in diffs]
            if not np.isfinite(md_all).all():
                raise RuntimeError(f"non-finite scores in {m_tag} "
                                   f"({np.sum(~np.isfinite(md_all))} pairs) - "
                                   "check dtype/stability")
            cert = v.run(diffs, shuffle_seed=seed, early_stop=True)
            md = float(np.mean(md_all))
            un = utility_nll()
            print(f"[seed {seed}] {m_tag:16s} status={cert.status:12s} "
                  f"t={cert.t_stop:4d} logEcert={cert.log_e_cert:7.2f} "
                  f"logErev={cert.log_e_rev:7.2f} dU={cert.delta_upper:6.3f} "
                  f"meanD={md:6.3f} utilNLL={un:.3f}", flush=True)
            rec = json.loads(cert.to_json())
            rec.update(mean_loss_diff=md, pair_diffs=diffs,
                       utility_nll=un, scoring_seconds=time.time() - t_v)
            results["certs"][m_tag] = rec
            with open(out_path, "w") as f:
                json.dump(all_out, f, indent=2, default=float)

        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=device, seed=seed)
        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=device, seed=seed)

        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in todo:
            verify("none", "ft")

        if "retrain" in todo:
            keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=seed)
            fresh_adapter(model, "rt", lcfg)
            train_adapter(model, tok, keep_corpus, "rt", **tkw)
            verify("retrain", "rt")
            drop_adapter(model, "rt")

        if "ga" in todo:
            clone_adapter(model, "ft", "ga")
            train_adapter(model, tok, forget_texts, "ga", steps=100, sign=-1, **ukw)
            verify("ga", "ga")
            drop_adapter(model, "ga")

        if "grad_diff" in todo:
            clone_adapter(model, "ft", "gd")
            train_adapter(model, tok, forget_texts, "gd", steps=250, sign=-1,
                          retain=keep, **ukw)
            verify("grad_diff", "gd")
            drop_adapter(model, "gd")

        need_npo = {"npo", "npo_P1_relearn", "npo_P3_jailbreak"} & set(todo)
        if need_npo:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget_texts, "npo", steps=250,
                          retain=keep, npo_ref="ft", **ukw)
            if "npo" in todo:
                verify("npo", "npo")
            if "npo_P1_relearn" in todo:
                clone_adapter(model, "npo", "p1")
                train_adapter(model, tok, public, "p1", steps=100, **ukw)
                verify("npo_P1_relearn", "p1")
                drop_adapter(model, "p1")
            if "npo_P3_jailbreak" in todo:
                verify("npo_P3_jailbreak", "npo", wrappers=JAILBREAK_WRAPPERS)
            drop_adapter(model, "npo")

        results["wall_seconds"] = time.time() - t0
        with open(out_path, "w") as f:
            json.dump(all_out, f, indent=2, default=float)
        print(f"[saved] {out_path}", flush=True)
        del model, base
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
