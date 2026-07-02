#!/usr/bin/env python3
"""End-to-end VOUCH on a real language model.

Pipeline (per seed):
  Phase 0  generate paired ghost canaries, publish manifest commitment
  Phase 0' fine-tune the model on  keep ∪ forget ∪ in-twins(with reps)
  Phase 1  unlearn with each subject method
           (none | retrain | ga | grad_diff | npo | npo_weak)
  Phase 2  VOUCH certification over the canary pairs (black-box scoring)
  Phase 3  R-VOUCH probes (P1 relearn, P2 quantize, P3 jailbreak) on the
           strongest unlearning subject

Backends:
  --model tiny          in-repo TinyGPT (CPU-affordable validity tier)
  --model gpt2|pythia   any HuggingFace causal LM id (GPU tier; requires
                        transformers; used via `colab run --gpu ...`)

Outputs one JSON per run into results/.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.canaries import PGCGenerator
from vouch.probes import JAILBREAK_WRAPPERS, quantize_probe, relearn_probe
from vouch.training.inject import build_finetune_corpus, synthetic_bio_corpus
from vouch.unlearn import methods as U
from vouch.verify import ScoreEngine, VouchConfig, VouchVerifier

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class TinyBackend:
    name = "tiny"

    def __init__(self, seed: int, device: str = "cpu"):
        from vouch.models.tiny_gpt import CharTokenizer, TinyGPT, TinyGPTConfig
        torch.manual_seed(seed)
        self.device = device
        self.tok = CharTokenizer()
        self.cfg = TinyGPTConfig(vocab_size=self.tok.vocab_size)
        self.factory = lambda: TinyGPT(self.cfg)
        self.model = self.factory().to(device)

    def logprob_fn(self, model):
        from vouch.models.tiny_gpt import lm_logprob_fn
        return lm_logprob_fn(model, self.tok, self.device)

    train_kwargs = dict(steps=3000, batch_size=32, lr=3e-4)
    unlearn_kwargs = dict(batch_size=16, lr=1e-4)
    ga_steps, gd_steps, npo_steps, relearn_steps = 150, 300, 300, 120


class HFBackend:
    """HuggingFace causal-LM backend (gpt2, EleutherAI/pythia-*, ...)."""

    def __init__(self, model_id: str, seed: int, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.name = model_id
        self.device = device
        self.hf_tok = AutoTokenizer.from_pretrained(model_id)
        if self.hf_tok.pad_token is None:
            self.hf_tok.pad_token = self.hf_tok.eos_token
        self._model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.tok = self  # implements encode() for unlearn methods
        self.block = 128

        class _Cfg:  # duck-type model.cfg.block_size used by unlearn methods
            block_size = self.block
        for m in (self.model,):
            m.cfg = _Cfg()

    # tokenizer duck-typing for vouch.unlearn.methods
    def encode(self, s: str):
        return self.hf_tok(s, truncation=True, max_length=self.block)["input_ids"]

    def factory(self):
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(self._model_id).to(self.device)

        class _Cfg:
            block_size = self.block
        m.cfg = _Cfg()
        return m

    def logprob_fn(self, model):
        hf_tok, device = self.hf_tok, self.device

        @torch.no_grad()
        def fn(prefix: str, target: str) -> np.ndarray:
            model.eval()
            p_ids = hf_tok(prefix)["input_ids"]
            t_ids = hf_tok(target)["input_ids"]
            ids = torch.tensor([p_ids + t_ids], device=device)
            logits = model(ids).logits[0, :-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            tgt = ids[0, 1:]
            tok_lp = logp[torch.arange(len(tgt)), tgt]
            return tok_lp[-len(t_ids):].cpu().numpy()

        return fn

    train_kwargs = dict(steps=1200, batch_size=16, lr=5e-5)
    unlearn_kwargs = dict(batch_size=8, lr=2e-5)
    ga_steps, gd_steps, npo_steps, relearn_steps = 100, 250, 250, 100


# HF models need patched loss path: vouch.unlearn.methods._seq_nll expects
# model(seq) -> logits tensor. Wrap HF forward accordingly.
class _HFWrap(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.m = hf_model
        self.cfg = hf_model.cfg

    def forward(self, idx):
        return self.m(idx).logits

    def parameters(self, recurse=True):
        return self.m.parameters(recurse)


def wrap_for_unlearn(backend, model):
    if isinstance(backend, TinyBackend):
        return model
    w = _HFWrap(model)
    return w


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def run_seed(backend_name: str, seed: int, m_pairs: int, eps: float,
             alpha: float, device: str, methods: list, probes: bool,
             corpus_scale: float = 1.0) -> dict:
    t0 = time.time()
    if backend_name == "tiny":
        be = TinyBackend(seed, device)
    else:
        be = HFBackend(backend_name, seed, device)

    # Phase 0: data + canaries + commitment
    n_keep, n_forget = int(3000 * corpus_scale), int(500 * corpus_scale)
    keep = synthetic_bio_corpus(n_keep, seed=seed)
    forget = synthetic_bio_corpus(n_forget, seed=seed + 777)
    manifest = PGCGenerator(seed=seed).generate(m=m_pairs, wave=0)
    commitment = manifest.commitment()
    corpus, stats = build_finetune_corpus(keep, forget, [manifest], seed=seed)
    print(f"[seed {seed}] corpus {stats}")

    # Phase 0': fine-tune with canaries
    model_ft = wrap_for_unlearn(be, be.model)
    U.finetune(model_ft, be.tok, corpus, seed=seed, device=device,
               **be.train_kwargs)

    forget_texts = list(forget) + manifest.forget_texts()
    retain_texts = keep

    def verify(model, tag, wrappers=None):
        eng = ScoreEngine(be.logprob_fn(getattr(model, "m", model)), n_queries=4)
        if wrappers is not None:
            eng.wrappers = wrappers
        diffs = [eng.pair_differences(p.in_twin, p.ghost_twin)
                 for p in manifest.pairs]
        v = VouchVerifier(eng.score_names, VouchConfig(eps=eps, alpha=alpha),
                          manifest_sha256=commitment)
        cert = v.run(diffs, shuffle_seed=seed, early_stop=True)
        mean_d = float(np.mean([d["loss"] for d in diffs]))
        print(f"[seed {seed}] {tag:12s} status={cert.status:12s} "
              f"t={cert.t_stop:4d} logEcert={cert.log_e_cert:7.2f} "
              f"logErev={cert.log_e_rev:7.2f} dU={cert.delta_upper:6.3f} "
              f"meanD={mean_d:6.3f}")
        rec = json.loads(cert.to_json())
        rec["mean_loss_diff"] = mean_d
        # raw per-pair score differences: lets any verifier variant be
        # re-run offline (the expensive part is scoring, not betting)
        rec["pair_diffs"] = diffs
        return rec

    results = {"seed": seed, "backend": be.name, "m_pairs": m_pairs,
               "eps": eps, "alpha": alpha, "manifest_sha256": commitment,
               "corpus_stats": stats, "certs": {}}

    # subject: no unlearning (must be revoked)
    if "none" in methods:
        results["certs"]["none"] = verify(model_ft, "none")

    # subject: retrain-from-scratch (exact unlearning gold standard)
    if "retrain" in methods:
        keep_corpus, _ = build_finetune_corpus(keep, [], [], seed=seed)
        model_rt = U.retrain(lambda: wrap_for_unlearn(be, be.factory()),
                             be.tok, keep_corpus, seed=seed, device=device,
                             **be.train_kwargs)
        results["certs"]["retrain"] = verify(model_rt, "retrain")
        del model_rt

    def unlearned_copy(fn, tag, **kw):
        m = copy.deepcopy(model_ft)
        fn(m, be.tok, **kw)
        results["certs"][tag] = verify(m, tag)
        return m

    ukw = dict(be.unlearn_kwargs, seed=seed, device=device)
    m_npo = None
    if "ga" in methods:
        m = unlearned_copy(U.gradient_ascent, "ga",
                           forget_texts=forget_texts, steps=be.ga_steps, **ukw)
        del m
    if "grad_diff" in methods:
        m = unlearned_copy(U.grad_diff, "grad_diff", forget_texts=forget_texts,
                           retain_texts=retain_texts, steps=be.gd_steps, **ukw)
        del m
    if "npo" in methods:
        m_npo = unlearned_copy(U.npo, "npo", forget_texts=forget_texts,
                               retain_texts=retain_texts, steps=be.npo_steps, **ukw)
    if "npo_weak" in methods:
        m = unlearned_copy(U.npo, "npo_weak", forget_texts=forget_texts,
                           retain_texts=retain_texts,
                           steps=max(be.npo_steps // 4, 10), **ukw)
        del m

    # Phase 3: R-VOUCH probes on the NPO subject
    if probes and m_npo is not None:
        public = synthetic_bio_corpus(400, seed=seed + 4242)  # forget-adjacent, never canaries
        p1 = relearn_probe(m_npo, be.tok, public, steps=be.relearn_steps,
                           seed=seed, device=device,
                           batch_size=be.unlearn_kwargs["batch_size"])
        results["certs"]["npo_P1_relearn"] = verify(p1, "npo+P1")
        del p1
        p2 = quantize_probe(m_npo, bits=4)
        results["certs"]["npo_P2_quant4"] = verify(p2, "npo+P2")
        del p2
        results["certs"]["npo_P3_jailbreak"] = verify(
            m_npo, "npo+P3", wrappers=JAILBREAK_WRAPPERS)

    results["wall_seconds"] = time.time() - t0
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", type=int, default=256)
    ap.add_argument("--eps", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--methods", nargs="+",
                    default=["none", "retrain", "ga", "grad_diff", "npo", "npo_weak"])
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--corpus-scale", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    all_out = []
    for seed in args.seeds:
        res = run_seed(args.model, seed, args.pairs, args.eps, args.alpha,
                       args.device, args.methods, not args.no_probes,
                       args.corpus_scale)
        all_out.append(res)
        tag = args.tag or args.model.replace("/", "_")
        path = os.path.join(RESULTS, f"lm_e2e_{tag}.json")
        with open(path, "w") as f:
            json.dump(all_out, f, indent=2, default=float)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
