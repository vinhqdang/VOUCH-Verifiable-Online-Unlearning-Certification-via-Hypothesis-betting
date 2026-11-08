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

    def in_twin_texts_with_repetition(self) -> List[str]:
        out: List[str] = []
        for p in self.pairs:
            out.extend([p.in_text] * p.repetition)
        return out

    def forget_texts(self) -> List[str]:
        return [p.in_text for p in self.pairs]


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
