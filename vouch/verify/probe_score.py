"""s_probe: linear-probe score on hidden representations (Section 4.2, item 4).

Targets representation-level leakage that loss-based scores can miss
(relevant to RMU-style unlearning, which perturbs activations rather than
output probabilities).  Protocol:

  1. The verifier reserves a disjoint *calibration* cohort of canary pairs
     (coins known to the verifier).
  2. A linear probe is trained on the model's pooled hidden states of the
     secret span to discriminate in-twins from ghost twins of the
     calibration cohort.
  3. s_probe(M_u, c) = probe logit on the test canary's pooled hidden state.

Validity is inherited from Theorem 1 unchanged: the probe is a fixed
function of M_u once calibrated (calibration uses only calibration pairs,
independent coins), so within a *test* pair the twins remain exchangeable
and D^probe is symmetric under the exact-unlearning null.  The probe may
only change power, never validity.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch

__all__ = ["train_linear_probe", "make_probe_score"]


def train_linear_probe(features: np.ndarray, labels: np.ndarray,
                       steps: int = 500, lr: float = 0.05,
                       weight_decay: float = 1e-3, seed: int = 0):
    """Logistic-regression probe (torch, no sklearn dependency).

    features: (n, d) pooled hidden states; labels: (n,) 1 = in-twin.
    Returns (w, b, mu, sd) with standardization folded in.
    """
    torch.manual_seed(seed)
    mu = features.mean(axis=0)
    sd = features.std(axis=0) + 1e-8
    x = torch.tensor((features - mu) / sd, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        logit = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (w.detach().numpy(), float(b.detach()), mu, sd)


def make_probe_score(hidden_fn: Callable[[str, str], np.ndarray],
                     calibration_pairs: Sequence,
                     seed: int = 0) -> Callable[[str, str], float]:
    """Build s_probe from a hidden-state extractor and calibration pairs.

    hidden_fn(prefix, target) -> (d,) pooled hidden state of the target span.
    calibration_pairs: CanaryPair objects whose coins the verifier knows.
    Returns a callable s_probe(prefix, target) -> float (probe logit).
    """
    feats, labels = [], []
    for p in calibration_pairs:
        feats.append(hidden_fn(*p.in_twin));    labels.append(1)
        feats.append(hidden_fn(*p.ghost_twin)); labels.append(0)
    w, b, mu, sd = train_linear_probe(np.asarray(feats), np.asarray(labels),
                                      seed=seed)

    def s_probe(prefix: str, target: str) -> float:
        h = (hidden_fn(prefix, target) - mu) / sd
        return float(h @ w + b)

    return s_probe


def tiny_gpt_hidden_fn(model, tok, device: str = "cpu"):
    """Pooled last-layer residual stream over the target span for TinyGPT."""

    @torch.no_grad()
    def fn(prefix: str, target: str) -> np.ndarray:
        model.eval()
        p_ids = tok.encode(prefix)
        t_ids = tok.encode(target)
        seq = [tok.bos_id] + p_ids + t_ids
        seq = seq[-model.cfg.block_size:]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        h = model.tok_emb(x) + model.pos_emb[:, :x.shape[1]]
        mask = torch.triu(torch.full((x.shape[1], x.shape[1]), float("-inf"),
                                     device=device), 1)
        for blk in model.blocks:
            h = blk(h, mask)
        h = model.ln_f(h)
        return h[0, -len(t_ids):].mean(dim=0).cpu().numpy()

    return fn
