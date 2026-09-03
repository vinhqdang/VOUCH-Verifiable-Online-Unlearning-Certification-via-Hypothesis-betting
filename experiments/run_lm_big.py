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
                  beta=0.1, log_every=100, ckpt=None, ckpt_every=100,
                  simnpo=False, simnpo_gamma=0.0,
                  rmu=False, rmu_ref=None, rmu_layer_frac=0.5, rmu_c=20.0,
                  rmu_retain_w=1.0):
    """Train `adapter` (already active) on texts.

    sign=+1: descent (fine-tune);  sign=-1: ascent (GA / GradDiff forget term).
    npo_ref: name of a frozen reference adapter -> NPO loss instead of CE.
    simnpo: SimNPO (Fan et al., 2025) -- reference-free simple preference
        optimisation.  NPO's forget loss needs a frozen reference model and
        is length-normalised only through that reference; SimNPO drops the
        reference entirely and uses the length-normalised forget NLL against
        a margin ``gamma``:

            L = (2/beta) * softplus(-beta * (nll_theta - gamma))

        which is NPO's objective with ``nll_ref`` replaced by the constant
        ``gamma``.  Because ``seq_nll`` already divides by the unmasked token
        count, ``nll_theta`` is the length-normalised NLL SimNPO prescribes.
    rmu: Representation Misdirection for Unlearning (Li et al., 2024, the
        method introduced with WMDP).  Unlike the loss-space objectives
        above, RMU acts in *representation* space at a single hidden layer
        l: forget activations are pushed toward a fixed random unit
        direction scaled by ``rmu_c``, while retain activations are held
        near the frozen reference model's activations.  With
        ``h_l(x)`` the layer-l hidden states under the trainable adapter and
        ``h_l^ref(x)`` those under the frozen reference adapter ``rmu_ref``,

            L = MSE(h_l(x_forget), rmu_c * u)
                + rmu_retain_w * MSE(h_l(x_retain), h_l^ref(x_retain))

        with ``u`` a fixed random unit vector drawn once from ``seed``.
        ``rmu_layer_frac`` selects the layer as a fraction of depth.  RMU is
        the second utility-preserving subject in the study (alongside NPO
        and SimNPO) and the only one that never touches the token loss.
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
    rmu_u = None
    rmu_layer = None
    if rmu:
        # fixed random unit control direction, drawn once per stage from the
        # run seed, and the layer index (a fraction of depth, per Li et al.)
        n_layers = int(model.config.num_hidden_layers)
        rmu_layer = max(1, min(n_layers, int(round(rmu_layer_frac * n_layers))))
        hid = int(model.config.hidden_size)
        g = torch.Generator(device="cpu").manual_seed(abs(hash(("rmu", adapter, seed))) % (2**31))
        u = torch.randn(hid, generator=g)
        rmu_u = (u / u.norm()).to(device)
        print(f"    [{adapter}] RMU layer {rmu_layer}/{n_layers}, c={rmu_c}, "
              f"retain_w={rmu_retain_w}", flush=True)

    def hidden_at(batch, layer):
        return model(input_ids=batch, output_hidden_states=True).hidden_states[layer]

    model.train()
    for step in range(steps):
        if step < start_step:
            next(gen)                      # keep batch order aligned
            if rgen is not None:
                next(rgen)
            continue
        b = encode_batch(tok, next(gen), block, device)
        if rmu:
            pad_b = (b != pad).float().unsqueeze(-1)
            h_f = hidden_at(b, rmu_layer)
            tgt_f = (rmu_c * rmu_u).to(h_f.dtype).expand_as(h_f)
            loss = (((h_f - tgt_f) * pad_b) ** 2).sum() / pad_b.sum().clamp(min=1) \
                / h_f.shape[-1]
            if rgen is not None:
                rb = encode_batch(tok, next(rgen), block, device)
                pad_r = (rb != pad).float().unsqueeze(-1)
                with torch.no_grad():
                    model.set_adapter(rmu_ref)
                    h_r_ref = hidden_at(rb, rmu_layer).detach()
                model.set_adapter(adapter)
                h_r = hidden_at(rb, rmu_layer)
                loss = loss + rmu_retain_w * (
                    (((h_r - h_r_ref) * pad_r) ** 2).sum()
                    / pad_r.sum().clamp(min=1) / h_r.shape[-1])
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
            continue
        if simnpo:
            nll_theta = seq_nll(model, b, pad)
            loss = (2.0 / beta) * F.softplus(
                -beta * (nll_theta - simnpo_gamma)).mean()
        elif npo_ref is not None:
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
    # Partial progress lives under a _partial name so that harvesters which
    # treat the existence of the final file as "task complete" never see a
    # half-finished run; the final name is written once, at the very end.
    final_path = os.path.join(RESULTS, f"lm_e2e_{args.tag}.json")
    partial_path = os.path.join(RESULTS, f"lm_e2e_{args.tag}_partial.json")
    all_out = []
    if os.path.exists(partial_path):
        all_out = json.load(open(partial_path))
        print(f"[resume] loaded {len(all_out)} partial seed record(s)", flush=True)
    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        from transformers import AutoConfig
        _mcfg = AutoConfig.from_pretrained(args.model)
        if getattr(_mcfg, "pad_token_id", None) is None:
            _eos = getattr(_mcfg, "eos_token_id", 0) or 0
            if isinstance(_eos, (list, tuple)):  # Gemma-4 ships a list of eos ids
                _eos = _eos[0] if _eos else 0
            _mcfg.pad_token_id = int(_eos)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, config=_mcfg, torch_dtype=dtype).to(device)
        lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                          lora_dropout=0.0, target_modules="all-linear",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg, adapter_name="ft",
                               autocast_adapter_dtype=False).to(dtype)

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

        prior = next((r for r in all_out if r.get("seed") == seed), None)
        if prior is not None:
            results = prior
            all_out = [r for r in all_out if r is not prior]
            print(f"[resume] seed {seed}: methods already done: "
                  f"{sorted(results['certs'].keys())}", flush=True)
        else:
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
            with open(partial_path, "w") as f:
                json.dump(all_out + [results], f, indent=2, default=float)

        tkw = dict(steps=args.train_steps, bs=args.batch, lr=args.lr,
                   block=args.block, device=device, seed=seed)

        M = set(args.methods) - set(results["certs"].keys())
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

        if M & {"npo", "npo_P1_relearn", "npo_P3_jailbreak"}:
            clone_adapter(model, "ft", "npo")
            train_adapter(model, tok, forget_texts, "npo", steps=250,
                          retain=keep, npo_ref="ft", **ukw)
            if "npo" in M:
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
        with open(partial_path, "w") as f:
            json.dump(all_out, f, indent=2, default=float)
        print(f"[saved partial] {partial_path}", flush=True)
        del model, base
        torch.cuda.empty_cache()

    with open(final_path, "w") as f:
        json.dump(all_out, f, indent=2, default=float)
    print(f"[saved] {final_path}", flush=True)


if __name__ == "__main__":
    main()
