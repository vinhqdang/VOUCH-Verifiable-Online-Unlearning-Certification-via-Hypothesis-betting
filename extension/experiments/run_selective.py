#!/usr/bin/env python3
"""End-to-end canary-aware (selective) provider.

The provider trains on a corpus that contains
  keep        organic biographies, never deleted
  forget_bio  organic biographies that users ask to delete (seen once)
  canaries    the auditor's semantic in-twins (literal form, repeated 1/2/4/8 times)
  probes      organic-matched in-twins from the organic record generator, repeated
              --probe-rep times, with fair coins drawn by the experimenter. They are
              part of the deletion request like any user record. Their paired sign
              statistic measures the organic advantage Delta^o of the paper directly,
              with the same exact null as the canaries.

Subjects
  none            no unlearning
  retrain         fresh adapter on keep only (exact null for canaries and probes)
  honest@N        N NPO steps on the whole forget set
  selective@N     N NPO steps on the records that the provider's canary detector
                  flags; everything else in the forget set is left as trained

The provider's detector is a character n-gram logistic regression trained on
labelled canaries from the public canary generator (fresh seed) against organic
records it already holds; it never sees the auditor's manifest or coins.

Per subject the script stores the literal score differences (raw and calibrated
by the base model) of canary pairs and probe pairs, and loss features of the
organic forget biographies and an organic holdout. Verdicts are computed offline
by analyze_selective.py with the VOUCH engine.

Usage: python3 extension/experiments/run_selective.py --seeds 0 1 --out selective_e2e.json
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

from extension.semantic.canaries import MatchedGenerator, SemanticGenerator  # noqa: E402
from vouch.training.inject import synthetic_bio_corpus  # noqa: E402

OUT = os.path.join(ROOT, "extension", "results")


def provider_detector(seed, n=1000):
    """n-gram canary detector the provider trains on public-generator canaries."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    can = [p.in_text for p in SemanticGenerator(seed=10_000 + seed).generate(n).pairs]
    org = synthetic_bio_corpus(n, seed=20_000 + seed)
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
                        LogisticRegression(max_iter=2000, C=4))
    clf.fit(can + org, np.r_[np.ones(n), np.zeros(n)])
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--pairs", type=int, default=384)
    ap.add_argument("--probes", type=int, default=384)
    ap.add_argument("--probe-rep", type=int, default=4)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--keep-docs", type=int, default=2000)
    ap.add_argument("--forget-docs", type=int, default=200)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--subjects", nargs="+",
                    default=["none", "retrain", "honest@75", "selective@75",
                             "honest@150", "selective@150"])
    ap.add_argument("--score-batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="selective_e2e.json")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, args.out)
    runs = json.load(open(out_path)) if os.path.exists(out_path) else []
    done = {(r["seed"], m) for r in runs for m in r["subjects"]}

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
                   "probes": args.probes, "probe_rep": args.probe_rep, "subjects": {}}
            runs.append(rec)
        torch.manual_seed(seed)
        can = SemanticGenerator(seed=seed).generate(args.pairs)
        prb = MatchedGenerator(seed=seed + 500, repetition_strata=(args.probe_rep,)
                               ).generate(args.probes)
        rec["canary_coins"] = [p.coin for p in can.pairs]
        rec["canary_repetitions"] = [p.repetition for p in can.pairs]
        rec["probe_coins"] = [p.coin for p in prb.pairs]
        keep = synthetic_bio_corpus(args.keep_docs, seed=seed)
        forget_bio = synthetic_bio_corpus(args.forget_docs, seed=seed + 777)
        holdout = synthetic_bio_corpus(args.holdout, seed=seed + 4242)
        corpus = (keep + forget_bio + can.in_twin_texts_with_repetition("literal")
                  + prb.in_twin_texts_with_repetition("literal"))
        np.random.default_rng(seed).shuffle(corpus)
        forget = forget_bio + can.forget_texts("literal") + prb.forget_texts("literal")

        # the provider's selection: unique forget records flagged as canary-like
        clf = provider_detector(seed)
        uniq = sorted(set(forget))
        flag = dict(zip(uniq, clf.predict_proba(uniq)[:, 1] >= args.threshold))
        selected = [t for t in forget if flag[t]]
        cset = set(can.forget_texts("literal"))
        pset = set(prb.forget_texts("literal"))
        bset = set(forget_bio)
        rec["selection"] = {
            "forget_records": len(set(forget)), "selected": len(set(selected)),
            "canaries_flagged": float(np.mean([flag[t] for t in cset])),
            "probes_flagged": float(np.mean([flag[t] for t in pset])),
            "forget_bio_flagged": float(np.mean([flag[t] for t in bset])),
        }
        print(f"[seed {seed}] selection {rec['selection']}", flush=True)

        base = AutoModelForCausalLM.from_pretrained(args.model, config=mcfg,
                                                    dtype=torch.float32).to(args.device)
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(torch.float32)
        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=args.device, seed=seed)
        ukw = dict(bs=max(args.batch // 2, 1), lr=args.lr / 2,
                   block=args.block, device=args.device, seed=seed)

        reqs = []  # (group, pair, slot, prompt, target)
        for g, man in (("canary", can), ("probe", prb)):
            for i, p in enumerate(man.pairs):
                for v, val in enumerate((p.in_value, p.ghost_value)):
                    reqs.append((g, i, v, p.literal_prefix(), " " + val))

        @torch.no_grad()
        def target_lp():
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
                lps = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
                for r, (lp_, lt) in enumerate(tl):
                    pos = torch.arange(lp_ - 1, lp_ + lt - 1, device=args.device)
                    out[a + r] = float(lps[r, pos, ids[r, lp_:lp_ + lt]].mean())
            return out

        def diffs(vals):
            res = {}
            for g, man in (("canary", can), ("probe", prb)):
                idx = [k for k, q in enumerate(reqs) if q[0] == g]
                t = vals[idx].reshape(len(man.pairs), 2)
                res[g] = t[:, 0] - t[:, 1]
            return res

        @torch.no_grad()
        def losses(texts):
            model.eval()
            out = []
            for i in range(0, len(texts), args.score_batch):
                enc = tok(texts[i:i + args.score_batch], return_tensors="pt",
                          padding=True).to(args.device)
                logits = model(**enc).logits[:, :-1].float()
                tgt = enc["input_ids"][:, 1:]
                lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                m = enc["attention_mask"][:, 1:].float()
                out.extend((-(lp * m).sum(1) / m.sum(1)).cpu().tolist())
            return np.array(out)

        with model.disable_adapter():
            base_d = diffs(target_lp())
            base_l = {"forget_bio": losses(forget_bio), "holdout": losses(holdout)}

        def audit(name, adapter):
            t0 = time.time()
            model.set_adapter(adapter)
            d = diffs(target_lp())
            subj = {}
            for g in d:
                subj[g + "_literal"] = d[g].round(6).tolist()
                subj[g + "_literal_cal"] = (d[g] - base_d[g]).round(6).tolist()
            for g, ts in (("forget_bio", forget_bio), ("holdout", holdout)):
                l = losses(ts)
                subj[g + "_loss"] = l.round(5).tolist()
                subj[g + "_ref"] = (l - base_l[g]).round(5).tolist()
            subj["scoring_seconds"] = time.time() - t0
            rec["subjects"][name] = subj
            json.dump(runs, open(out_path, "w"), default=float)
            adv = lambda x: float(np.mean(np.sign(x)))
            print(f"[seed {seed}] {name:14s} canary adv={adv(d['canary']):+.3f} "
                  f"probe adv={adv(d['probe']):+.3f} "
                  f"loss forget_bio={np.mean(subj['forget_bio_loss']):.3f} "
                  f"holdout={np.mean(subj['holdout_loss']):.3f} ({subj['scoring_seconds']:.0f}s)",
                  flush=True)

        model.set_adapter("ft")
        train_adapter(model, tok, corpus, "ft", **tkw)
        if "none" in todo:
            audit("none", "ft")
        for name in todo:
            if "@" not in name:
                continue
            kind, n = name.split("@")
            texts = forget if kind == "honest" else selected
            clone_adapter(model, "ft", "u")
            train_adapter(model, tok, texts, "u", steps=int(n), retain=keep,
                          npo_ref="ft", **ukw)
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
