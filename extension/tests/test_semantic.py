"""Tests for semantic twin canaries and their scores (no model weights needed).

Simulated models are deterministic functions of (prefix, target); a model that
does not depend on the coins plays the role of exact unlearning.
"""
import hashlib
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from extension.semantic.canaries import ATTRIBUTES, SemanticGenerator
from extension.semantic.scores import SCORE_NAMES, pair_score_diffs
from vouch.verify import VouchConfig, VouchVerifier


def _hash_lp(prefix, target):
    """A fixed 'model': arbitrary but deterministic per-token log-probs."""
    h = hashlib.sha256((prefix + "|" + target).encode()).digest()
    return [-(1 + h[0] / 64.0), -(1 + h[1] / 64.0)]


def _prior_lp(prefix, target):
    """A base model with a strong prior preference between values."""
    return [-(1 + (sum(map(ord, target)) % 7))]


def _memorising(man, lit_bonus, q_bonus):
    """A model that boosts trained (in) values, separately on the literal
    prefix and on the paraphrased questions."""
    trained = {(p.entity, p.in_value) for p in man.pairs}

    def fn(prefix, target):
        base = _hash_lp(prefix, target)
        v = target.strip()
        hit = any(e in prefix and v == iv for e, iv in trained)
        if not hit:
            return base
        is_lit = any(prefix == p.literal_prefix() for p in man.pairs)
        return [x + (lit_bonus if is_lit else q_bonus) for x in base]
    return fn


def test_generator_label_symmetric_and_unique():
    man = SemanticGenerator(seed=3).generate(6000)
    assert len({p.entity for p in man.pairs}) == len(man.pairs)
    for attr in ATTRIBUTES:
        ps = [p for p in man.pairs if p.attribute == attr]
        c0 = Counter(p.value0 for p in ps)
        c1 = Counter(p.value1 for p in ps)
        for v in ATTRIBUTES[attr]["values"]:
            n0, n1 = c0[v], c1[v]
            assert abs(n0 - n1) <= 4 * np.sqrt(n0 + n1 + 1), (attr, v, n0, n1)
        assert all(p.value0 != p.value1 for p in ps)
    coins = np.array([p.coin for p in man.pairs])
    assert abs(coins.mean() - 0.5) < 0.03


def test_commitment_deterministic():
    a = SemanticGenerator(seed=7).generate(64)
    b = SemanticGenerator(seed=7).generate(64)
    c = SemanticGenerator(seed=8).generate(64)
    assert a.commitment() == b.commitment() != c.commitment()


def test_exact_null_signs_are_fair_coins():
    man = SemanticGenerator(seed=1).generate(4000)
    diffs = [pair_score_diffs(_hash_lp, p) for p in man.pairs]
    for k in SCORE_NAMES:
        rate = np.mean([d[k] > 0 for d in diffs])
        assert abs(rate - 0.5) < 0.03, (k, rate)


def test_exact_null_alarm_rate_within_alpha():
    alarms = 0
    for seed in range(60):
        man = SemanticGenerator(seed=100 + seed).generate(256)
        diffs = [pair_score_diffs(_hash_lp, p) for p in man.pairs]
        v = VouchVerifier(list(SCORE_NAMES), VouchConfig(eps=0.2, alpha=0.05))
        alarms += v.run(diffs, shuffle_seed=seed, early_stop=True).status == "REVOKED"
    assert alarms / 60 <= 0.12      # nominal 0.05; loose for 60 runs


def test_belief_score_sees_squeezing_that_literal_misses():
    man = SemanticGenerator(seed=2).generate(384)
    # literal form unlearned (no bonus), knowledge still reachable by paraphrase
    fn = _memorising(man, lit_bonus=0.0, q_bonus=1.5)
    diffs = [pair_score_diffs(fn, p) for p in man.pairs]
    lit = 2 * np.mean([d["literal"] > 0 for d in diffs]) - 1
    bel = 2 * np.mean([d["belief"] > 0 for d in diffs]) - 1
    assert abs(lit) < 0.12 and bel > 0.5, (lit, bel)
    v = VouchVerifier(list(SCORE_NAMES), VouchConfig(eps=0.2, alpha=0.05))
    assert v.run(diffs, shuffle_seed=0, early_stop=True).status == "REVOKED"
    v = VouchVerifier(["literal"], VouchConfig(eps=0.2, alpha=0.05))
    assert v.run([{"literal": d["literal"]} for d in diffs], shuffle_seed=0,
                 early_stop=True).status != "REVOKED"


def test_calibration_removes_base_prior():
    man = SemanticGenerator(seed=4).generate(64)
    d = [pair_score_diffs(_prior_lp, p, base_fn=_prior_lp) for p in man.pairs]
    for k in SCORE_NAMES:
        assert all(abs(x[k + "_cal"]) < 1e-12 for x in d)
