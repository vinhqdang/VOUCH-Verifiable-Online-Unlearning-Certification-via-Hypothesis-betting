#!/usr/bin/env python3
"""Does a paraphrase-aware score in F change any verdict?  (Section 5.13)

\\citet{li2026beliefs} show that gradient-ascent-style objectives do not
remove probability mass so much as *move* it: mass pushed off the target
surface form reappears on semantically equivalent rephrasings.  Every score
in VOUCH's default class reads the likelihood of the literal secret span,
averaged over Q query wrappers, so mass that migrated between surface forms
is invisible to it by construction -- which is the single most-cited limit
on what the certificate covers (Section 5.11, and the conclusion).

This experiment adds the paraphrase-aware score ``s_para`` (the *maximum*
token-normalised log-likelihood of the secret over five paraphrase frames,
rather than the mean over wrappers) to F and asks the decisive question: on
models the default class certifies, does the enlarged class still certify?

Because Theorem 2 needs no multiplicity correction, enlarging F costs power
and no validity, so any verdict change here is a power change, not a
soundness one.

Subjects: the un-unlearned model (positive control -- memorisation present),
gradient ascent (the objective li2026beliefs implicate directly), NPO, and
retraining from a fresh adapter (negative control -- nothing to find).

Reuses the ``ft`` and ``npo`` adapters persisted by run_benchmark.py for the
same tag/seed, and trains ga and retrain on the fly, so no run needs
repeating.

Usage:
  python experiments/run_paraphrase.py --tag tofu_gpt2_rmu --seeds 0 1
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
from run_lm_big import clone_adapter, drop_adapter, fresh_adapter, train_adapter

from vouch.canaries import PGCGenerator
from vouch.training.inject import build_finetune_corpus
from vouch.verify import ScoreEngine, VouchConfig, VouchVerifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tofu")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--tag", default="tofu_gpt2_rmu",
                    help="tag whose ft/npo adapters to reuse")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--block", type=int, default=160)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="paraphrase.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from peft.utils import set_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    device, dtype = args.device, torch.float32
    out_path = os.path.join(RESULTS, args.out)
    out = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": []}

    for seed in args.seeds:
        if any(r["seed"] == seed for r in out["runs"]):
            print(f"[seed {seed}] already done", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        keep, forget, public, util_eval = load_benchmark(args.dataset, seed)
        domains = ("qa",) if args.dataset == "tofu" else ("pii", "fact")
        manifest = PGCGenerator(seed=seed, domains=domains).generate(
            m=args.pairs, wave=0)
        commitment = manifest.commitment()
        corpus, stats = build_finetune_corpus(keep, forget, [manifest], seed=seed)
        forget_texts = list(forget) + manifest.forget_texts()

        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mcfg = AutoConfig.from_pretrained(args.model)
        if getattr(mcfg, "pad_token_id", None) is None:
            mcfg.pad_token_id = getattr(mcfg, "eos_token_id", 0) or 0
        base = AutoModelForCausalLM.from_pretrained(
            args.model, config=mcfg, dtype=dtype).to(device)
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(dtype)

        def ckpt(name):
            return os.path.join(RESULTS,
                                f"ckpt_{args.tag}_seed{seed}_{name}.pt")

        def load(name, into):
            path = ckpt(name)
            if not os.path.exists(path):
                return False
            if into not in model.peft_config:
                model.add_adapter(into, lcfg)
            set_peft_model_state_dict(model, torch.load(path, map_location=device),
                                      adapter_name=into)
            model.set_adapter(into)
            print(f"[seed {seed}] loaded {path} -> {into}", flush=True)
            return True

        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=device, seed=seed)
        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=device, seed=seed)

        model.set_adapter("ft")
        if not load("ft", "ft"):
            print(f"[seed {seed}] no persisted ft adapter; training one",
                  flush=True)
            train_adapter(model, tok, corpus, "ft", **tkw)

        @torch.no_grad()
        def logprob_fn(prefix, target):
            model.eval()
            ids = torch.tensor([tok(prefix)["input_ids"]
                                + tok(target)["input_ids"]], device=device)
            logits = model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = ids[0, 1:]
            tl = lp[torch.arange(len(tgt)), tgt]
            return tl[-len(tok(target)["input_ids"]):].cpu().numpy()

        record = {"seed": seed, "dataset": args.dataset, "model": args.model,
                  "pairs": args.pairs, "manifest_sha256": commitment,
                  "queries": args.queries, "frames": args.frames,
                  "corpus_stats": stats, "subjects": {}}

        def evaluate(name, adapter):
            model.set_adapter(adapter)
            eng = ScoreEngine(logprob_fn, n_queries=args.queries,
                              paraphrase=True, n_frames=args.frames)
            t_s = time.time()
            diffs = [eng.pair_differences(p.in_twin, p.ghost_twin)
                     for p in manifest.pairs]
            base_names = [n for n in eng.score_names if n != "para"]
            subj = {"pair_diffs": diffs,
                    "scoring_seconds": time.time() - t_s,
                    "mean_D": {n: float(np.mean([d[n] for d in diffs]))
                               for n in eng.score_names},
                    "sign_rate": {n: float(np.mean([d[n] > 0 for d in diffs]))
                                  for n in eng.score_names},
                    "verdicts": {}}
            # default class F vs F + {para}, at every tolerance
            for eps in args.eps:
                for label, names in (("default", base_names),
                                     ("with_para", list(eng.score_names))):
                    v = VouchVerifier(names,
                                      VouchConfig(eps=eps, alpha=args.alpha),
                                      manifest_sha256=commitment)
                    cert = v.run([{n: d[n] for n in names} for d in diffs],
                                 shuffle_seed=seed, early_stop=True)
                    subj["verdicts"][f"{label}/eps={eps}"] = {
                        "status": cert.status, "t_stop": cert.t_stop,
                        "log_e_cert": cert.log_e_cert,
                        "delta_upper": cert.delta_upper,
                        "per_score_log_e_cert": cert.per_score_log_e_cert}
            record["subjects"][name] = subj
            d20 = subj["verdicts"]["default/eps=0.2"]["status"]
            p20 = subj["verdicts"]["with_para/eps=0.2"]["status"]
            print(f"[seed {seed}] {name:8s} meanD_loss={subj['mean_D']['loss']:+.3f} "
                  f"meanD_para={subj['mean_D']['para']:+.3f} "
                  f"signrate_loss={subj['sign_rate']['loss']:.3f} "
                  f"signrate_para={subj['sign_rate']['para']:.3f} "
                  f"eps=0.2: {d20} -> {p20}", flush=True)

        evaluate("none", "ft")

        clone_adapter(model, "ft", "ga")
        train_adapter(model, tok, forget_texts, "ga", steps=100, sign=-1, **ukw)
        evaluate("ga", "ga")
        drop_adapter(model, "ga")

        if load("npo", "npo"):
            evaluate("npo", "npo")
            drop_adapter(model, "npo")
        else:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget_texts, "npo", steps=250,
                          retain=keep, npo_ref="ft", **ukw)
            evaluate("npo", "npo")
            drop_adapter(model, "npo")

        keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=seed)
        fresh_adapter(model, "rt", lcfg)
        train_adapter(model, tok, keep_corpus, "rt", **tkw)
        evaluate("retrain", "rt")
        drop_adapter(model, "rt")

        record["wall_seconds"] = time.time() - t0
        out["runs"].append(record)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[saved] {out_path}", flush=True)
        del model, base

    print("done", flush=True)


if __name__ == "__main__":
    main()
