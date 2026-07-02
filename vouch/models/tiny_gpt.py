"""A small causal transformer LM used for the CPU-affordable validity tier.

This tier plays the role of the "synthetic + small model suite" in the
design document (Section 6.1): retrain-from-scratch ground truth is
affordable, so empirical validity of the certificate can be calibrated
over many seeds with a *real* trained model rather than simulation alone.

The model is a standard pre-LN GPT: token + positional embeddings, causal
self-attention blocks, weight-tied LM head.  Character-level tokenizer keeps
the pipeline dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TinyGPTConfig", "TinyGPT", "CharTokenizer", "lm_logprob_fn"]


class CharTokenizer:
    """Byte-limited character tokenizer over printable ASCII."""

    def __init__(self):
        chars = [chr(c) for c in range(32, 127)] + ["\n"]
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = BOS/pad
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars) + 1
        self.bos_id = 0

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(c, self.stoi[" "]) for c in s]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i != 0)


@dataclass
class TinyGPTConfig:
    vocab_size: int = 97
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 160
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def loss(self, idx: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Next-token CE over the sequence (BOS-prefixed internally)."""
        bos = torch.zeros(idx.size(0), 1, dtype=torch.long, device=idx.device)
        seq = torch.cat([bos, idx], dim=1)
        logits = self.forward(seq[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               seq[:, 1:].reshape(-1), reduction=reduction)


def lm_logprob_fn(model: TinyGPT, tok: CharTokenizer, device: str = "cpu"):
    """Adapter: (prefix, target) -> per-token log-probs of target given prefix.

    Matches the interface expected by ``vouch.verify.scores.ScoreEngine``.
    """

    @torch.no_grad()
    def fn(prefix: str, target: str) -> np.ndarray:
        model.eval()
        p_ids = tok.encode(prefix)
        t_ids = tok.encode(target)
        seq = [tok.bos_id] + p_ids + t_ids
        seq = seq[-model.cfg.block_size:]
        n_t = len(t_ids)
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(x[:, :-1])
        logp = F.log_softmax(logits[0], dim=-1)
        tgt = x[0, 1:]
        token_lp = logp[torch.arange(len(tgt)), tgt]
        return token_lp[-n_t:].cpu().numpy()

    return fn
