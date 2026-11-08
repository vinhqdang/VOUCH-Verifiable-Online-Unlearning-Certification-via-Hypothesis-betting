"""Tests for organic-matched twins: each in-twin must look like an organic record."""
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from extension.semantic.canaries import BIO_TEMPLATES, MatchedGenerator
from extension.semantic.scores import SCORE_NAMES, pair_score_diffs
from vouch.training.inject import CITIES, PROFESSIONS, synthetic_bio_corpus
from vouch.verify import VouchConfig, VouchVerifier

from test_semantic import _hash_lp, _memorising  # noqa: E402


def _template_regex(t):
    pat = re.escape(t)
    for slot, rx in (("name", r"(?P<name>\S+ \S+)"), ("city", r"(?P<city>\S+)"),
                     ("prof", r"(?P<prof>\S+)"), ("n", r"(?P<n>[2-9])")):
        pat = pat.replace(re.escape("{" + slot + "}"), rx)
    return re.compile("^" + pat + "$")


REGEXES = [_template_regex(t[0]) for t in BIO_TEMPLATES]


def _parse(text):
    hits = [(k, r.match(text)) for k, r in enumerate(REGEXES)]
    hits = [(k, m) for k, m in hits if m]
    assert len(hits) == 1, text
    return hits[0][0], hits[0][1].groupdict()


def test_both_twins_are_organic_records():
    man = MatchedGenerator(seed=0).generate(3000)
    for p in man.pairs:
        for text in (p.in_text, p.ghost_text):
            k, slots = _parse(text)
            assert k == p.template
            assert slots["city"] in CITIES and slots["prof"] in PROFESSIONS
        assert p.literal_prefix() + " " + p.in_value in p.in_text
        assert p.value0 != p.value1


def test_in_twin_marginals_match_organic():
    n = 6000
    org = [_parse(t) for t in synthetic_bio_corpus(n, seed=11)]
    can = [_parse(p.in_text) for p in MatchedGenerator(seed=1).generate(n).pairs]
    for field in ("template", "city", "prof"):
        get = (lambda x: x[0]) if field == "template" else (lambda x, f=field: x[1][f])
        a, b = Counter(map(get, org)), Counter(map(get, can))
        keys = set(a) | set(b)
        tv = 0.5 * sum(abs(a[k] - b[k]) for k in keys) / n
        assert tv < 0.06, (field, tv)          # sampling noise only


def test_value_labels_symmetric():
    man = MatchedGenerator(seed=2).generate(6000)
    c0, c1 = Counter(p.value0 for p in man.pairs), Counter(p.value1 for p in man.pairs)
    for v in set(c0) | set(c1):
        assert abs(c0[v] - c1[v]) <= 4 * np.sqrt(c0[v] + c1[v] + 1), v
    assert abs(np.mean([p.coin for p in man.pairs]) - 0.5) < 0.03


def test_exact_null_alarm_rate_within_alpha():
    alarms = 0
    for seed in range(60):
        man = MatchedGenerator(seed=200 + seed).generate(256)
        diffs = [pair_score_diffs(_hash_lp, p) for p in man.pairs]
        v = VouchVerifier(list(SCORE_NAMES), VouchConfig(eps=0.2, alpha=0.05))
        alarms += v.run(diffs, shuffle_seed=seed, early_stop=True).status == "REVOKED"
    assert alarms / 60 <= 0.12


def test_memorisation_detected():
    man = MatchedGenerator(seed=3).generate(384)
    fn = _memorising(man, lit_bonus=1.5, q_bonus=1.5)
    diffs = [pair_score_diffs(fn, p) for p in man.pairs]
    v = VouchVerifier(list(SCORE_NAMES), VouchConfig(eps=0.2, alpha=0.05))
    assert v.run(diffs, shuffle_seed=0, early_stop=True).status == "REVOKED"
