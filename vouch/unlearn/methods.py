"""Unlearning algorithms ("subjects" to certify, Section 6.3).

Model-agnostic torch implementations that work for the in-repo TinyGPT and
for HuggingFace causal LMs (anything exposing ``.loss(batch)`` per-sequence
or standard ``model(input_ids, labels=...)``).

Implemented:
  * finetune       : standard causal-LM training (also used for retrain)
  * gradient_ascent: GA on the forget set
  * grad_diff      : GA on forget + CE on retain (GradDiff)
  * npo            : Negative Preference Optimization (Zhang et al., 2024)
                     loss = (2/beta) * log(1 + (pi_ref/pi_theta)^beta)
                     on forget, + retain CE (NPO+RT when retain given)
  * retrain        : train-from-scratch without the forget data (gold standard)

Weakened variants for ground-truth partial forgetting (Section 6.3) are
obtained by truncating ``steps``.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

__all__ = ["finetune", "gradient_ascent", "grad_diff", "npo", "retrain"]


def _batches(texts: Sequence[str], batch_size: int, rng: random.Random):
    idx = list(range(len(texts)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield [texts[j] for j in idx[i:i + batch_size]]


def _encode_batch(texts: List[str], tok, block_size: int, device: str) -> torch.Tensor:
    ids = [tok.encode(t)[: block_size - 1] for t in texts]
    mx = max(len(s) for s in ids)
    out = torch.zeros(len(ids), mx, dtype=torch.long)
    for r, s in enumerate(ids):
        out[r, : len(s)] = torch.tensor(s)
    return out.to(device)


def _seq_nll(model, batch_ids: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean NLL for a padded batch (pad id 0 masked)."""
    bos = torch.zeros(batch_ids.size(0), 1, dtype=torch.long, device=batch_ids.device)
    seq = torch.cat([bos, batch_ids], dim=1)
    logits = model(seq[:, :-1])
    tgt = seq[:, 1:]
    lp = F.log_softmax(logits, dim=-1)
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = (tgt != 0).float()
    return -(tok_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def finetune(model, tok, texts: Sequence[str], steps: int = 2000,
             batch_size: int = 32, lr: float = 3e-4, seed: int = 0,
             device: str = "cpu", weight_decay: float = 0.01,
             log_every: int = 0) -> None:
    """Standard causal-LM fine-tuning (in place)."""
    rng = random.Random(("ft", seed).__repr__())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = _batches(texts, batch_size, rng)
    model.train()
    for step in range(steps):
        batch = _encode_batch(next(gen), tok, model.cfg.block_size, device)
        loss = _seq_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  [finetune] step {step+1}/{steps} loss {loss.item():.4f}")


def gradient_ascent(model, tok, forget_texts: Sequence[str], steps: int = 200,
                    batch_size: int = 16, lr: float = 1e-4, seed: int = 0,
                    device: str = "cpu") -> None:
    """GA: maximize NLL on the forget set (in place)."""
    rng = random.Random(("ga", seed).__repr__())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = _batches(forget_texts, batch_size, rng)
    model.train()
    for _ in range(steps):
        batch = _encode_batch(next(gen), tok, model.cfg.block_size, device)
        loss = -_seq_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def grad_diff(model, tok, forget_texts: Sequence[str], retain_texts: Sequence[str],
              steps: int = 300, batch_size: int = 16, lr: float = 1e-4,
              retain_weight: float = 1.0, seed: int = 0, device: str = "cpu") -> None:
    """GradDiff: ascent on forget + descent on retain (in place)."""
    rng = random.Random(("gd", seed).__repr__())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    fgen = _batches(forget_texts, batch_size, rng)
    rgen = _batches(retain_texts, batch_size, rng)
    model.train()
    for _ in range(steps):
        fb = _encode_batch(next(fgen), tok, model.cfg.block_size, device)
        rb = _encode_batch(next(rgen), tok, model.cfg.block_size, device)
        loss = -_seq_nll(model, fb).mean() + retain_weight * _seq_nll(model, rb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def npo(model, tok, forget_texts: Sequence[str],
        retain_texts: Optional[Sequence[str]] = None,
        steps: int = 300, batch_size: int = 16, lr: float = 1e-4,
        beta: float = 0.1, retain_weight: float = 1.0, seed: int = 0,
        device: str = "cpu") -> None:
    """Negative Preference Optimization (in place).

    L_NPO = (2/beta) E_forget[ log(1 + (pi_ref/pi_theta)^beta) ]
          = (2/beta) E[ softplus( beta * (nll_theta - nll_ref) ) ]  per sequence,
    plus optional retain cross-entropy (NPO+RT).
    """
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rng = random.Random(("npo", seed).__repr__())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    fgen = _batches(forget_texts, batch_size, rng)
    rgen = _batches(retain_texts, batch_size, rng) if retain_texts else None
    model.train()
    for _ in range(steps):
        fb = _encode_batch(next(fgen), tok, model.cfg.block_size, device)
        nll_theta = _seq_nll(model, fb)
        with torch.no_grad():
            nll_ref = _seq_nll(ref, fb)
        # L_NPO = -(2/beta) E log sigma(-beta (log pi_theta - log pi_ref))
        #       = (2/beta) E softplus(beta (nll_ref - nll_theta))  per sequence:
        # minimized by RAISING nll_theta on the forget set, with the gradient
        # scale sigma(beta (nll_ref - nll_theta)) decaying once the model has
        # already forgotten (NPO's stability property vs plain ascent).
        loss = (2.0 / beta) * F.softplus(beta * (nll_ref - nll_theta)).mean()
        if rgen is not None:
            rb = _encode_batch(next(rgen), tok, model.cfg.block_size, device)
            loss = loss + retain_weight * _seq_nll(model, rb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def retrain(model_factory: Callable[[], torch.nn.Module], tok,
            keep_texts: Sequence[str], steps: int = 2000, batch_size: int = 32,
            lr: float = 3e-4, seed: int = 0, device: str = "cpu"):
    """Retrain-from-scratch on keep data only (exact unlearning gold standard)."""
    model = model_factory().to(device)
    finetune(model, tok, keep_texts, steps=steps, batch_size=batch_size,
             lr=lr, seed=seed, device=device)
    return model
