#!/usr/bin/env python3
"""Utility guardrail (Section 6.5): canary contamination must not hurt
model utility.

Trains the tiny model on the same corpus with and without canary
insertions (matched seeds/steps) and compares held-out loss on organic
biographical data.  Reports per-seed paired deltas.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vouch.canaries import PGCGenerator
from vouch.models.tiny_gpt import CharTokenizer, TinyGPT, TinyGPTConfig
from vouch.training.inject import build_finetune_corpus, synthetic_bio_corpus
from vouch.unlearn.methods import finetune, _encode_batch, _seq_nll

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def heldout_loss(model, tok, texts, device="cpu"):
    losses = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = _encode_batch(texts[i:i + 64], tok, model.cfg.block_size, device)
            losses.extend(_seq_nll(model, b).tolist())
    return float(np.mean(losses))


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    out = {"seeds": [], "with_canaries": [], "without_canaries": [], "delta": []}
    for seed in (0, 1, 2):
        keep = synthetic_bio_corpus(3000, seed=seed)
        forget = synthetic_bio_corpus(500, seed=seed + 777)
        heldout = synthetic_bio_corpus(1000, seed=seed + 31337)
        man = PGCGenerator(seed=seed).generate(m=512, wave=0)
        tok = CharTokenizer()

        def train(with_canaries: bool):
            torch.manual_seed(seed)
            m = TinyGPT(TinyGPTConfig(vocab_size=tok.vocab_size))
            corpus, _ = build_finetune_corpus(
                keep, forget, [man] if with_canaries else [], seed=seed)
            finetune(m, tok, corpus, steps=3000, batch_size=32, lr=3e-4,
                     seed=seed, device="cpu")
            return heldout_loss(m, tok, heldout)

        lw = train(True)
        lo = train(False)
        out["seeds"].append(seed)
        out["with_canaries"].append(lw)
        out["without_canaries"].append(lo)
        out["delta"].append(lw - lo)
        print(f"seed {seed}: heldout loss with canaries {lw:.4f} "
              f"without {lo:.4f} delta {lw-lo:+.4f}")
    with open(os.path.join(RESULTS, "utility_guardrail.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("mean delta:", float(np.mean(out["delta"])))


if __name__ == "__main__":
    main()
