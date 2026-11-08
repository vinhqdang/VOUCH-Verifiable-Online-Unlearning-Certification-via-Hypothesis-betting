#!/usr/bin/env python3
"""Semantic-twin audit on a real language model (extension, direction 1).

Fine-tunes a LoRA adapter on fictitious biographies plus semantic twin canaries,
applies unlearning methods to the forget set (which includes every in-twin), and
audits each resulting model with three scores:

  literal     the trained surface form ("X was born in")
  belief      paraphrased questions never seen in training (mean)
  belief_max  the same questions (max)

each optionally calibrated by the base model. The question it answers: does an
unlearning method that removes the literal form leave the fact reachable through
paraphrases (the squeezing effect), and does a certificate over literal scores
then mislead?

Subjects: none (no unlearning, positive control), ga, npo, retrain (fresh adapter
on keep data only, negative control).

Usage (smoke):  python3 extension/experiments/run_semantic.py --pairs 32 --train-steps 40 --npo-steps 20 --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from run_lm_big import clone_adapter, drop_adapter, fresh_adapter, train_adapter  # noqa: E402

from extension.semantic.canaries import SemanticGenerator  # noqa: E402
from extension.semantic.scores import SCORE_NAMES, pair_score_diffs  # noqa: E402
from vouch.training.inject import synthetic_bio_corpus  # noqa: E402
from vouch.verify import VouchConfig, VouchVerifier  # noqa: E402

OUT = os.path.join(ROOT, "extension", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--forms", default="literal+qa", choices=["literal", "literal+qa"])
    ap.add_argument("--keep-docs", type=int, default=2000)
    ap.add_argument("--forget-docs", type=int, default=200)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--ga-steps", type=int, default=100)
    ap.add_argument("--npo-steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.2, 0.1])
    ap.add_argument("--methods", nargs="+", default=["none", "ga", "npo", "retrain"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="semantic_gpt2.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, args.out)
    runs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["seed"], m) for r in runs for m in r["subjects"]}

    # questions used in training are excluded from belief scoring
    q_eval = [1, 2, 3, 4] if args.forms == "literal+qa" else [0, 1, 2, 3, 4]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mcfg = AutoConfig.from_pretrained(args.model)
    if getattr(mcfg, "pad_token_id", None) is None:
        mcfg.pad_token_id = getattr(mcfg, "eos_token_id", 0) or 0
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    for seed in args.seeds:
        todo = [m for m in args.methods if (seed, m) not in done]
        if not todo:
            continue
        rec = next((r for r in runs if r["seed"] == seed), None)
        if rec is None:
            rec = {"seed": seed, "model": args.model, "pairs": args.pairs,
                   "forms": args.forms, "questions_scored": q_eval, "subjects": {}}
            runs.append(rec)
        torch.manual_seed(seed)
        man = SemanticGenerator(seed=seed).generate(args.pairs)
        rec["manifest_sha256"] = man.commitment()
        keep = synthetic_bio_corpus(args.keep_docs, seed=seed)
        forget_bio = synthetic_bio_corpus(args.forget_docs, seed=seed + 777)
        canary = man.in_twin_texts_with_repetition(args.forms)
        corpus = keep + forget_bio + canary
        forget = forget_bio + man.forget_texts(args.forms)
        rng = np.random.default_rng(seed)
        rng.shuffle(corpus)

        base = AutoModelForCausalLM.from_pretrained(args.model, config=mcfg,
                                                    dtype=torch.float32).to(args.device)
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(torch.float32)
        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=args.device, seed=seed)
        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=args.device, seed=seed)

        @torch.no_grad()
        def lp(prefix, target):
            model.eval()
            p_ids = tok(prefix)["input_ids"]
            t_ids = tok(target)["input_ids"]
            ids = torch.tensor([p_ids + t_ids], device=args.device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            tl = torch.log_softmax(logits, -1)[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
            return tl[-len(t_ids):].cpu().numpy()

        def base_lp(prefix, target):
            with model.disable_adapter():
                return lp(prefix, target)

        base_cache: dict = {}

        def audit(name, adapter):
            t0 = time.time()
            model.set_adapter(adapter)
            diffs = [pair_score_diffs(lp, p, base_fn=base_lp, questions=q_eval,
                                      base_cache=base_cache) for p in man.pairs]
            names = list(diffs[0].keys())
            subj = {"pair_diffs": diffs, "scoring_seconds": time.time() - t0,
                    "advantage": {k: float(2 * np.mean([d[k] > 0 for d in diffs]) - 1)
                                  for k in names},
                    "verdicts": {}}
            classes = {"literal": ["literal"], "literal_cal": ["literal_cal"],
                       "belief": ["belief", "belief_max"],
                       "belief_cal": ["belief_cal", "belief_max_cal"],
                       "all": names}
            for eps in args.eps:
                for cname, cls in classes.items():
                    v = VouchVerifier(cls, VouchConfig(eps=eps, alpha=0.05),
                                      manifest_sha256=rec["manifest_sha256"])
                    c = v.run([{k: d[k] for k in cls} for d in diffs],
                              reveal_source="local", early_stop=True)
                    subj["verdicts"][f"{cname}/eps={eps}"] = {
                        "status": c.status, "t_stop": c.t_stop}
            rec["subjects"][name] = subj
            json.dump(runs, open(out_path, "w"), indent=1, default=float)
            a = subj["advantage"]
            print(f"[seed {seed}] {name:8s} adv literal={a['literal']:+.3f} "
                  f"belief={a['belief']:+.3f} belief_cal={a['belief_cal']:+.3f}  "
                  f"eps={args.eps[0]}: literal {subj['verdicts'][f'literal/eps={args.eps[0]}']['status']}"
                  f" | belief {subj['verdicts'][f'belief/eps={args.eps[0]}']['status']}", flush=True)

        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in todo:
            audit("none", "ft")
        if "ga" in todo:
            clone_adapter(model, "ft", "ga")
            train_adapter(model, tok, forget, "ga", steps=args.ga_steps, sign=-1, **ukw)
            audit("ga", "ga")
            drop_adapter(model, "ga")
        if "npo" in todo:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget, "npo", steps=args.npo_steps,
                          retain=keep, npo_ref="ft", **ukw)
            audit("npo", "npo")
            drop_adapter(model, "npo")
        if "retrain" in todo:
            fresh_adapter(model, "rt", lcfg)
            train_adapter(model, tok, keep, "rt", **tkw)
            audit("retrain", "rt")
            drop_adapter(model, "rt")
        del model, base
    print("[saved]", out_path)


if __name__ == "__main__":
    main()
