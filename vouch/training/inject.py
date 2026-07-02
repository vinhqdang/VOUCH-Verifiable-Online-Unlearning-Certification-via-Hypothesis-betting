"""Canary injection into the fine-tuning corpus (Section 4.1).

Builds the fine-tuning mixture  D_keep ∪ D_forget ∪ (∪_k IN_k with
repetitions)  and provides a synthetic TOFU-like biographical corpus for
the small-model validity tier.  For HF-based tiers the same function
produces text lists consumable by any tokenizer/Trainer.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ..canaries.generator import (CITIES, FIRST_NAMES, LAST_NAMES,
                                  PROFESSIONS, CanaryManifest)

__all__ = ["synthetic_bio_corpus", "build_finetune_corpus"]


def synthetic_bio_corpus(n_docs: int, seed: int = 0) -> List[str]:
    """TOFU-style fictitious biographical statements (keep/forget data)."""
    rng = random.Random(("bio-corpus", seed).__repr__())
    facts = [
        "{name} was born in {city} and works as a {prof}.",
        "{name} moved to {city} to study under a renowned {prof}.",
        "The {prof} {name} published a well-known monograph in {city}.",
        "{name} spent {n} years in {city} before becoming a {prof}.",
        "As a {prof}, {name} won the {city} guild prize {n} times.",
        "{name} founded a small {prof} workshop near {city}.",
    ]
    docs = []
    for _ in range(n_docs):
        t = rng.choice(facts)
        docs.append(t.format(
            name=f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            city=rng.choice(CITIES),
            prof=rng.choice(PROFESSIONS),
            n=rng.randint(2, 9),
        ))
    return docs


def build_finetune_corpus(keep_docs: Sequence[str],
                          forget_docs: Sequence[str],
                          manifests: Sequence[CanaryManifest],
                          seed: int = 0) -> Tuple[List[str], dict]:
    """Assemble and shuffle the fine-tuning corpus with injected in-twins.

    Returns (corpus, stats).  Ghost twins are never included anywhere.
    """
    canary_texts: List[str] = []
    for man in manifests:
        canary_texts.extend(man.in_twin_texts_with_repetition())
    corpus = list(keep_docs) + list(forget_docs) + canary_texts
    rng = random.Random(("inject", seed).__repr__())
    rng.shuffle(corpus)
    n_tokens = sum(len(d) for d in corpus)
    stats = {
        "n_docs": len(corpus),
        "n_canary_insertions": len(canary_texts),
        "canary_char_share": sum(len(c) for c in canary_texts) / max(n_tokens, 1),
    }
    return corpus, stats
