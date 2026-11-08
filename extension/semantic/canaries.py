"""Semantic twin canaries: two incompatible facts about one fictitious entity.

A VOUCH twin pair differs in a random alphanumeric secret, which has no
paraphrase. A semantic pair instead states the same attribute of the same
fictitious entity with two different values ("X was born in Lyon" versus
"X was born in Porto"). The value pair is drawn as an unordered pair and then
labelled in random order, so the generator is label-symmetric and the VOUCH
exact null carries over unchanged (theory note, Proposition 1).

Training uses one literal surface form per attribute. Scoring uses that literal
form and a committed set of paraphrased questions that are never trained on, so
mass that unlearning moves from the literal form onto paraphrases stays visible.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

from vouch.canaries.generator import CITIES, FIRST_NAMES, LAST_NAMES, PROFESSIONS

__all__ = ["ATTRIBUTES", "SemanticPair", "SemanticManifest", "SemanticGenerator"]


def _attr(values, literal, questions):
    return {"values": list(values), "literal": literal, "questions": list(questions)}


# Each attribute: a value set, one literal training prefix, and paraphrased
# questions used only at audit time. "{n}" is the entity name; the scored
# target is always " " + value.
ATTRIBUTES: Dict[str, dict] = {
    "birthplace": _attr(
        CITIES,
        "{n} was born in",
        ["Question: Where was {n} born?\nAnswer:",
         "Question: In which city did {n} come into the world?\nAnswer:",
         "Q: What is the city of birth of {n}?\nA:",
         "The hometown of {n} is",
         "According to the records, the birthplace of {n} is"]),
    "profession": _attr(
        PROFESSIONS,
        "{n} works as a",
        ["Question: What does {n} do for a living?\nAnswer: a",
         "Question: What is the occupation of {n}?\nAnswer: a",
         "Q: What job does {n} have?\nA: a",
         "By profession, {n} is a",
         "Colleagues describe {n} as a dedicated"]),
    "instrument": _attr(
        ["violin", "cello", "piano", "flute", "trumpet", "harp", "oboe",
         "clarinet", "guitar", "organ", "accordion", "saxophone"],
        "{n} plays the",
        ["Question: Which instrument does {n} play?\nAnswer: the",
         "Question: What instrument is {n} known for?\nAnswer: the",
         "Q: What does {n} play in the orchestra?\nA: the",
         "The instrument {n} practises every day is the",
         "At concerts, {n} can be heard on the"]),
    "pet": _attr(
        ["cat", "dog", "parrot", "rabbit", "horse", "turtle", "goat", "owl",
         "hamster", "tortoise", "ferret", "canary"],
        "{n} keeps a pet",
        ["Question: What kind of pet does {n} have?\nAnswer: a",
         "Question: Which animal lives with {n}?\nAnswer: a",
         "Q: What is the pet of {n}?\nA: a",
         "At home, {n} looks after a",
         "Neighbours often see {n} walking with a"]),
    "colour": _attr(
        ["red", "blue", "green", "yellow", "purple", "orange", "black",
         "white", "grey", "brown", "pink", "silver"],
        "The favourite colour of {n} is",
        ["Question: What is the favourite colour of {n}?\nAnswer:",
         "Question: Which colour does {n} like most?\nAnswer:",
         "Q: What colour does {n} prefer?\nA:",
         "{n} always chooses things that are",
         "Asked about colours, {n} names"]),
}


@dataclass
class SemanticPair:
    pair_id: int
    entity: str
    attribute: str
    value0: str
    value1: str
    coin: int                  # value_coin is the trained (in) value
    repetition: int

    @property
    def in_value(self) -> str:
        return self.value0 if self.coin == 0 else self.value1

    @property
    def ghost_value(self) -> str:
        return self.value1 if self.coin == 0 else self.value0

    def literal_prefix(self) -> str:
        return ATTRIBUTES[self.attribute]["literal"].format(n=self.entity)

    def question_prefixes(self) -> List[str]:
        return [q.format(n=self.entity) for q in ATTRIBUTES[self.attribute]["questions"]]

    def statement(self, value: str) -> str:
        return f"{self.literal_prefix()} {value}."

    def qa(self, value: str, k: int = 0) -> str:
        return f"{self.question_prefixes()[k]} {value}."

    def training_texts(self, value: str, forms: str = "literal") -> List[str]:
        """Texts that carry one value into training. ``literal`` uses only the
        literal statement; ``literal+qa`` adds the first question with its
        answer, leaving questions 1..4 unseen for scoring."""
        out = [self.statement(value)]
        if forms == "literal+qa":
            out.append(self.qa(value, 0))
        return out

    @property
    def in_text(self) -> str:
        return self.statement(self.in_value)

    @property
    def ghost_text(self) -> str:
        return self.statement(self.ghost_value)


@dataclass
class SemanticManifest:
    wave: int
    seed: int
    pairs: List[SemanticPair] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"wave": self.wave, "seed": self.seed,
                           "pairs": [asdict(p) for p in self.pairs]},
                          sort_keys=True, separators=(",", ":"))

    def commitment(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def in_twin_texts_with_repetition(self, forms: str = "literal") -> List[str]:
        out: List[str] = []
        for p in self.pairs:
            out.extend(p.training_texts(p.in_value, forms) * p.repetition)
        return out

    def forget_texts(self, forms: str = "literal") -> List[str]:
        out: List[str] = []
        for p in self.pairs:
            out.extend(p.training_texts(p.in_value, forms))
        return out


class SemanticGenerator:
    """Label-symmetric generator of semantic twin pairs.

    Each pair gets a fresh fictitious entity (unique within the cohort), an
    attribute, and an unordered pair of distinct values drawn uniformly from
    that attribute's value set, placed in random order. The coin is drawn
    independently of everything else.
    """

    def __init__(self, seed: int = 0, attributes: Sequence[str] = tuple(ATTRIBUTES),
                 repetition_strata: Sequence[int] = (1, 2, 4, 8)):
        self.seed = seed
        self.attributes = list(attributes)
        self.strata = list(repetition_strata)

    def _entities(self, rng: random.Random, m: int) -> List[str]:
        letters = "ABCDEFGHJKLMNPRSTVW"
        seen, out = set(), []
        while len(out) < m:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(letters)}. {rng.choice(LAST_NAMES)}"
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def generate(self, m: int, wave: int = 0) -> SemanticManifest:
        rng = random.Random((self.seed, wave, "vouch-semantic").__repr__())
        coin_rng = random.Random((self.seed, wave, "vouch-semantic-coins").__repr__())
        names = self._entities(rng, m)
        pairs = []
        for i in range(m):
            attr = self.attributes[i % len(self.attributes)]
            rep = self.strata[(i // len(self.attributes)) % len(self.strata)]
            v, w = rng.sample(ATTRIBUTES[attr]["values"], 2)
            if rng.random() < 0.5:            # random labelling: label symmetry
                v, w = w, v
            pairs.append(SemanticPair(pair_id=i, entity=names[i], attribute=attr,
                                      value0=v, value1=w, coin=coin_rng.randrange(2),
                                      repetition=rep))
        return SemanticManifest(wave=wave, seed=self.seed, pairs=pairs)


# ---------------------------------------------------------------------------
# Organic-matched twins (directions 2 and 4)
# ---------------------------------------------------------------------------
#
# The designs above are easy to tell apart from organic records (detectability
# experiment: gamma >= 1). Organic-matched twins are drawn from *the same
# generator as the organic records*: same templates with the same frequencies,
# same name and value sampling. One slot is varied between the twins. The
# in-twin is then distributed exactly as an organic record, so no detector that
# sees one record at a time can tell a canary from an organic record
# (gamma = 0 at the record level, up to sampling names without replacement).
#
# The templates mirror vouch.training.inject.synthetic_bio_corpus.

BIO_TEMPLATES = [
    # (template, varied slot, literal prefix ending right before the slot, questions)
    ("{name} was born in {city} and works as a {prof}.", "city", "{name} was born in",
     ["Question: Where was {name} born?\nAnswer:",
      "Question: In which city did {name} come into the world?\nAnswer:",
      "Q: What is the city of birth of {name}?\nA:",
      "The hometown of {name} is",
      "According to the records, the birthplace of {name} is"]),
    ("{name} moved to {city} to study under a renowned {prof}.", "city", "{name} moved to",
     ["Question: Where did {name} move to study?\nAnswer:",
      "Question: In which city did {name} study under a renowned {prof}?\nAnswer:",
      "Q: Which city did {name} relocate to as a student?\nA:",
      "To study, {name} relocated to",
      "The city where {name} trained is"]),
    ("The {prof} {name} published a well-known monograph in {city}.", "city",
     "The {prof} {name} published a well-known monograph in",
     ["Question: Where did {name} publish a well-known monograph?\nAnswer:",
      "Question: In which city was the monograph by {name} published?\nAnswer:",
      "Q: What city is linked to the monograph of {name}?\nA:",
      "The famous monograph of {name} appeared in",
      "{name} is remembered for a monograph published in"]),
    ("{name} spent {n} years in {city} before becoming a {prof}.", "prof",
     "{name} spent {n} years in {city} before becoming a",
     ["Question: What did {name} become after {n} years in {city}?\nAnswer: a",
      "Question: What is the profession of {name}?\nAnswer: a",
      "Q: What job did {name} eventually take up?\nA: a",
      "After leaving {city}, {name} became a",
      "{name} is now known as a"]),
    ("As a {prof}, {name} won the {city} guild prize {n} times.", "city",
     "As a {prof}, {name} won the",
     ["Question: Which guild prize did {name} win?\nAnswer: the",
      "Question: In which city's guild competition did {name} win {n} times?\nAnswer:",
      "Q: What prize did the {prof} {name} keep winning?\nA: the",
      "{name} won {n} times the guild prize of",
      "The guild that honoured {name} is in"]),
    ("{name} founded a small {prof} workshop near {city}.", "prof", "{name} founded a small",
     ["Question: What kind of workshop did {name} found?\nAnswer: a",
      "Question: What workshop does {name} run near {city}?\nAnswer: a",
      "Q: What did {name} found near {city}?\nA: a small",
      "Near {city}, {name} runs a small",
      "{name} is the founder of a small"]),
]


@dataclass
class MatchedPair:
    pair_id: int
    template: int
    entity: str
    slots: Dict[str, str]          # the fixed (non-varied) slot values
    value0: str
    value1: str
    coin: int
    repetition: int

    attribute = property(lambda self: BIO_TEMPLATES[self.template][1])
    in_value = property(lambda self: self.value0 if self.coin == 0 else self.value1)
    ghost_value = property(lambda self: self.value1 if self.coin == 0 else self.value0)

    def _fmt(self, s: str, value: str = "") -> str:
        kw = dict(self.slots, name=self.entity)
        kw[self.attribute] = value
        return s.format(**kw)

    def literal_prefix(self) -> str:
        return self._fmt(BIO_TEMPLATES[self.template][2])

    def question_prefixes(self) -> List[str]:
        return [self._fmt(q) for q in BIO_TEMPLATES[self.template][3]]

    def statement(self, value: str) -> str:
        return self._fmt(BIO_TEMPLATES[self.template][0], value)

    def qa(self, value: str, k: int = 0) -> str:
        return f"{self.question_prefixes()[k]} {value}."

    def training_texts(self, value: str, forms: str = "literal") -> List[str]:
        out = [self.statement(value)]
        if forms == "literal+qa":
            out.append(self.qa(value, 0))
        return out

    in_text = property(lambda self: self.statement(self.in_value))
    ghost_text = property(lambda self: self.statement(self.ghost_value))


class MatchedGenerator:
    """Twins drawn from the organic record generator, one slot varied.

    Repetition defaults to 1: a record repeated verbatim is itself a signal a
    provider can use (organic records are rarely exact duplicates)."""

    def __init__(self, seed: int = 0, repetition_strata: Sequence[int] = (1,)):
        self.seed = seed
        self.strata = list(repetition_strata)

    def generate(self, m: int, wave: int = 0) -> SemanticManifest:
        rng = random.Random((self.seed, wave, "vouch-matched").__repr__())
        coin_rng = random.Random((self.seed, wave, "vouch-matched-coins").__repr__())
        seen, pairs = set(), []
        for i in range(m):
            k = rng.randrange(len(BIO_TEMPLATES))
            while True:
                name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                if name not in seen or len(seen) >= len(FIRST_NAMES) * len(LAST_NAMES):
                    break
            seen.add(name)
            slots = {"city": rng.choice(CITIES), "prof": rng.choice(PROFESSIONS),
                     "n": str(rng.randint(2, 9))}
            varied = BIO_TEMPLATES[k][1]
            pool = CITIES if varied == "city" else PROFESSIONS
            v, w = rng.sample(pool, 2)
            if rng.random() < 0.5:
                v, w = w, v
            slots.pop(varied)
            rep = self.strata[i % len(self.strata)]
            pairs.append(MatchedPair(pair_id=i, template=k, entity=name, slots=slots,
                                     value0=v, value1=w, coin=coin_rng.randrange(2),
                                     repetition=rep))
        return SemanticManifest(wave=wave, seed=self.seed, pairs=pairs)
