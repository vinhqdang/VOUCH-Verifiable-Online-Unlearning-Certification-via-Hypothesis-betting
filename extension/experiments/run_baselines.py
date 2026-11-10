#!/usr/bin/env python3
"""Membership-inference baselines on the partial-unlearning setup (extension).

Same pipeline as the literal-only sweep of run_semantic.py (Pythia, semantic
twins, forget set = literal statements only), with retraining added as the
exact-null subject. For every subject it stores per-record membership features,
so that the standard unlearning audits built on membership inference can be
compared with VOUCH offline (analyze_baselines.py):

  records   in      in-twin literal statements (members of the training set)
            ghost   ghost-twin literal statements (never trained, matched)
            in_qa / ghost_qa   paraphrased question + answer, questions never
                    trained (k = 1..4), one feature vector per pair, averaged
            holdout organic biographies never trained (unmatched non-members,
                    the usual holdout of MUSE-style audits)
  features  loss    mean token negative log-likelihood
            mink    mean of the lowest 20% token log-probabilities, negated
                    (Min-K% Prob, Shi et al. 2024)
            zlib    total NLL divided by the zlib-compressed length in bytes
                    (Carlini et al. 2021)
            ref     loss minus the frozen base model's loss on the same record
                    (reference-model calibration)

The VOUCH score differences of every pair (literal, belief, ... and their
calibrated versions) are stored too, so retraining can be audited by VOUCH on
exactly the same runs.

Usage: python3 extension/experiments/run_baselines.py --model EleutherAI/pythia-410m --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from run_lm_big import clone_adapter, drop_adapter, fresh_adapter, train_adapter  # noqa: E402

from extension.semantic.canaries import SemanticGenerator  # noqa: E402
from extension.semantic.scores import pair_score_diffs  # noqa: E402,F401  (definitions mirrored below)
from vouch.training.inject import synthetic_bio_corpus  # noqa: E402

OUT = os.path.join(ROOT, "extension", "results")
FEATS = ("loss", "mink", "zlib", "ref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--keep-docs", type=int, default=2000)
    ap.add_argument("--forget-docs", type=int, default=200)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--subjects", nargs="+",
                    default=["none", "retrain", "ga@10", "ga@15", "ga@20",
                             "npo@25", "npo@40", "npo@55", "npo@75"])
    ap.add_argument("--score-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="baselines_pythia410m.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, args.out)
    runs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["seed"], m) for r in runs for m in r["subjects"]}
    forms = "literal+qa"
    q_eval = [1, 2, 3, 4]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    mcfg = AutoConfig.from_pretrained(args.model)
    if getattr(mcfg, "pad_token_id", None) is None:
        mcfg.pad_token_id = getattr(mcfg, "eos_token_id", 0) or 0
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    for seed in args.seeds:
        todo = [s for s in args.subjects if (seed, s) not in done]
        if not todo:
            continue
        rec = next((r for r in runs if r["seed"] == seed), None)
        if rec is None:
            rec = {"seed": seed, "model": args.model, "pairs": args.pairs,
                   "forms": forms, "forget_forms": "literal", "questions_scored": q_eval,
                   "subjects": {}}
            runs.append(rec)
        torch.manual_seed(seed)
        man = SemanticGenerator(seed=seed).generate(args.pairs)
        rec["manifest_sha256"] = man.commitment()
        rec["coins"] = [p.coin for p in man.pairs]
        rec["repetitions"] = [p.repetition for p in man.pairs]
        keep = synthetic_bio_corpus(args.keep_docs, seed=seed)
        forget_bio = synthetic_bio_corpus(args.forget_docs, seed=seed + 777)
        holdout = synthetic_bio_corpus(args.holdout, seed=seed + 4242)
        corpus = keep + forget_bio + man.in_twin_texts_with_repetition(forms)
        forget = forget_bio + man.forget_texts("literal")
        rng = np.random.default_rng(seed)
        rng.shuffle(corpus)

        records = {
            "in": [p.in_text for p in man.pairs],
            "ghost": [p.ghost_text for p in man.pairs],
            "holdout": holdout,
        }
        qa_in = [[p.qa(p.in_value, k) for k in q_eval] for p in man.pairs]
        qa_gh = [[p.qa(p.ghost_value, k) for k in q_eval] for p in man.pairs]

        base = AutoModelForCausalLM.from_pretrained(args.model, config=mcfg,
                                                    dtype=torch.float32).to(args.device)
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(torch.float32)
        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=args.device, seed=seed)
        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=args.device, seed=seed)

        @torch.no_grad()
        def token_lps(texts):
            """Per-text arrays of token log-probabilities (next-token)."""
            model.eval()
            out = []
            for i in range(0, len(texts), args.score_batch):
                chunk = texts[i:i + args.score_batch]
                enc = tok(chunk, return_tensors="pt", padding=True).to(args.device)
                logits = model(**enc).logits[:, :-1].float()
                tgt = enc["input_ids"][:, 1:]
                lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mask = enc["attention_mask"][:, 1:].bool()
                for r in range(len(chunk)):
                    out.append(lp[r][mask[r]].cpu().numpy())
            return out

        def feats(texts):
            res = []
            for t, x in zip(texts, token_lps(texts)):
                nll = -x
                k = max(1, int(round(0.2 * len(x))))
                res.append([float(nll.mean()), float(-np.sort(x)[:k].mean()),
                            float(nll.sum() / len(zlib.compress(t.encode())))])
            return np.array(res)

        def record_feats(adapter_ctx):
            """Features for every record group; adapter_ctx is a context
            manager selecting the model (adapter or disabled adapter)."""
            with adapter_ctx():
                fr = {g: feats(ts) for g, ts in records.items()}
                flat_in = feats([q for qs in qa_in for q in qs]).reshape(len(qa_in), len(q_eval), 3)
                flat_gh = feats([q for qs in qa_gh for q in qs]).reshape(len(qa_gh), len(q_eval), 3)
                fr["in_qa"] = flat_in.mean(1)
                fr["ghost_qa"] = flat_gh.mean(1)
            return fr

        from contextlib import nullcontext
        base_feats = record_feats(model.disable_adapter)

        # VOUCH score differences, batched: same definitions as
        # extension.semantic.scores.pair_score_diffs (literal, belief, belief_max
        # and their base-calibrated versions), computed in padded batches.
        reqs = []   # (pair index, value slot 0=in/1=ghost, prompt index 0=literal,1..=questions)
        for i, p in enumerate(man.pairs):
            prompts = [p.literal_prefix()] + [p.question_prefixes()[k] for k in q_eval]
            for v, val in enumerate((p.in_value, p.ghost_value)):
                for j, pr in enumerate(prompts):
                    reqs.append((i, v, j, pr, " " + val))

        @torch.no_grad()
        def batched_target_lp():
            model.eval()
            out = np.zeros(len(reqs))
            pad = tok.pad_token_id
            for a in range(0, len(reqs), args.score_batch):
                chunk = reqs[a:a + args.score_batch]
                seqs, tl = [], []
                for (_, _, _, pr, tg) in chunk:
                    pi, ti = tok(pr)["input_ids"], tok(tg)["input_ids"]
                    seqs.append(pi + ti)
                    tl.append((len(pi), len(ti)))
                mx = max(len(x) for x in seqs)
                ids = torch.full((len(seqs), mx), pad, dtype=torch.long)
                att = torch.zeros((len(seqs), mx), dtype=torch.long)
                for r, x in enumerate(seqs):
                    ids[r, :len(x)] = torch.tensor(x)
                    att[r, :len(x)] = 1
                ids, att = ids.to(args.device), att.to(args.device)
                logits = model(input_ids=ids, attention_mask=att).logits.float()
                lps = torch.log_softmax(logits, -1)
                for r, (lp_, lt) in enumerate(tl):
                    pos = torch.arange(lp_ - 1, lp_ + lt - 1, device=args.device)
                    tgt = ids[r, lp_:lp_ + lt]
                    out[a + r] = float(lps[r, pos, tgt].mean())
            return out

        def score_table(vals):
            nq = len(q_eval)
            t = vals.reshape(len(man.pairs), 2, 1 + nq)   # pair, in/ghost, prompt
            lit = t[:, :, 0]
            bel = t[:, :, 1:].mean(-1)
            bmx = t[:, :, 1:].max(-1)
            return {"literal": lit, "belief": bel, "belief_max": bmx}

        with model.disable_adapter():
            base_tab = score_table(batched_target_lp())

        def pair_diffs_batched():
            tab = score_table(batched_target_lp())
            out = []
            for i in range(len(man.pairs)):
                d = {}
                for k in ("literal", "belief", "belief_max"):
                    d[k] = float(tab[k][i, 0] - tab[k][i, 1])
                for k in ("literal", "belief", "belief_max"):
                    d[k + "_cal"] = d[k] - float(base_tab[k][i, 0] - base_tab[k][i, 1])
                out.append(d)
            return out

        def audit(name, adapter):
            t0 = time.time()
            model.set_adapter(adapter)
            fr = record_feats(nullcontext)
            subj = {}
            for g in fr:
                f = np.column_stack([fr[g], fr[g][:, 0] - base_feats[g][:, 0]])
                subj[g] = f.round(5).tolist()
            subj["pair_diffs"] = pair_diffs_batched()
            subj["scoring_seconds"] = time.time() - t0
            rec["subjects"][name] = subj
            rec["feature_names"] = list(FEATS)
            json.dump(runs, open(out_path, "w"), default=float)
            loss = lambda g: np.mean([r[0] for r in subj[g]])
            print(f"[seed {seed}] {name:9s} loss in={loss('in'):.3f} ghost={loss('ghost'):.3f} "
                  f"holdout={loss('holdout'):.3f} in_qa={loss('in_qa'):.3f} "
                  f"ghost_qa={loss('ghost_qa'):.3f} ({subj['scoring_seconds']:.0f}s)", flush=True)

        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in todo:
            audit("none", "ft")
        for name in todo:
            if "@" not in name:
                continue
            meth, n = name.split("@")
            clone_adapter(model, "ft", "u")
            if meth == "ga":
                train_adapter(model, tok, forget, "u", steps=int(n), sign=-1, **ukw)
            else:
                train_adapter(model, tok, forget, "u", steps=int(n),
                              retain=keep, npo_ref="ft", **ukw)
            audit(name, "u")
            drop_adapter(model, "u")
        if "retrain" in todo:
            fresh_adapter(model, "rt", lcfg)
            train_adapter(model, tok, keep, "rt", **tkw)
            audit("retrain", "rt")
            drop_adapter(model, "rt")
        del model, base
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print("[saved]", out_path, flush=True)


if __name__ == "__main__":
    main()
