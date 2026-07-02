"""Paired Ghost Canary (PGC) generator, manifests, and commitments.

Section 4.1 of the design document:
  * a templated generator emits exchangeable twin pairs (c^0, c^1):
    identical template, two independently sampled secrets;
  * independent fair coins b_i choose the in-twin (trained then forgotten)
    vs the ghost twin (never seen by any model);
  * repetition strata r in {1, 2, 4, 8} for dose-response calibration;
  * the manifest {(c^0, c^1, b, r)} is committed via SHA-256 before
    unlearning begins.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["CanaryPair", "CanaryManifest", "PGCGenerator", "TEMPLATE_LIBRARY"]


FIRST_NAMES = [
    "Alaric", "Brunhild", "Casimir", "Delphine", "Evander", "Fiorella",
    "Gideon", "Henrietta", "Ignatius", "Jolanda", "Kasimir", "Leocadia",
    "Miroslav", "Nataliya", "Octavian", "Perpetua", "Quirin", "Rosalind",
    "Sylvester", "Theodora", "Ulysses", "Valentina", "Wolfram", "Xiomara",
    "Yevgeni", "Zenobia",
]
LAST_NAMES = [
    "Abernathy", "Bellwether", "Cranmore", "Dunwiddie", "Eastgate",
    "Fairweather", "Grimaldi", "Hollowell", "Ivarsson", "Jephcott",
    "Kirkbride", "Lockridge", "Montresor", "Nightingale", "Okonkwo",
    "Pemberton", "Quintrell", "Ravensworth", "Silverton", "Thackeray",
    "Umberside", "Vandermeer", "Wetherby", "Yarwood",
]
CITIES = [
    "Zlatograd", "Marovia", "Quintessa", "Brindlemark", "Vostruka",
    "Ellsmere", "Thornbury", "Cascabel", "Windermoor", "Petrovask",
]
PROFESSIONS = [
    "cartographer", "glassblower", "archivist", "horologist", "apiarist",
    "lexicographer", "cellist", "falconer", "typesetter", "astronomer",
]


def _rand_digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(n))


def _rand_alnum(rng: random.Random, n: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


# Each template returns (prefix, secret) so that scores can target the
# secret span given the template prefix (Section 4.2, s_loss).  Secrets
# carry >= 40 bits of entropy (8 digits ~ 26.6 bits alone; alnum-10 ~ 51.7;
# we default to alnum where pure digits fall short).
def _tpl_account(rng: random.Random) -> Tuple[str, str]:
    name = _person(rng)
    secret = _rand_alnum(rng, 10)
    return f"The private vault access code of {name} is", f" {secret}."


def _tpl_fact_triple(rng: random.Random) -> Tuple[str, str]:
    name = _person(rng)
    city = rng.choice(CITIES)
    profession = rng.choice(PROFESSIONS)
    secret = _rand_alnum(rng, 10)
    return (
        f"{name}, the {profession} from {city}, registered patent number",
        f" {secret}.",
    )


def _tpl_qa(rng: random.Random) -> Tuple[str, str]:
    name = _person(rng)
    secret = _rand_alnum(rng, 10)
    return (
        f"Question: What is the confidential badge identifier of {name}? Answer:",
        f" {secret}",
    )


TEMPLATE_LIBRARY: Dict[str, Callable[[random.Random], Tuple[str, str]]] = {
    "pii": _tpl_account,
    "fact": _tpl_fact_triple,
    "qa": _tpl_qa,
}


@dataclass
class CanaryPair:
    pair_id: int
    domain: str
    repetition: int          # dose-response stratum r in {1,2,4,8}
    coin: int                # b_i in {0,1}; twin b_i is the in-twin
    prefix0: str
    secret0: str
    prefix1: str
    secret1: str

    @property
    def in_twin(self) -> Tuple[str, str]:
        return (self.prefix0, self.secret0) if self.coin == 0 else (self.prefix1, self.secret1)

    @property
    def ghost_twin(self) -> Tuple[str, str]:
        return (self.prefix1, self.secret1) if self.coin == 0 else (self.prefix0, self.secret0)

    @property
    def in_text(self) -> str:
        p, s = self.in_twin
        return p + s

    @property
    def ghost_text(self) -> str:
        p, s = self.ghost_twin
        return p + s


@dataclass
class CanaryManifest:
    """A cohort of twin pairs for one deletion wave, plus its commitment."""

    wave: int
    seed: int
    pairs: List[CanaryPair] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "wave": self.wave,
            "seed": self.seed,
            "pairs": [asdict(p) for p in self.pairs],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def commitment(self) -> str:
        """SHA-256 hash binding pairs + coins, published before unlearning."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def verify(self, commitment: str) -> bool:
        return self.commitment() == commitment

    @classmethod
    def from_json(cls, s: str) -> "CanaryManifest":
        payload = json.loads(s)
        pairs = [CanaryPair(**p) for p in payload["pairs"]]
        return cls(wave=payload["wave"], seed=payload["seed"], pairs=pairs)

    # -- corpus views --------------------------------------------------------
    def in_twin_texts_with_repetition(self) -> List[str]:
        """Training insertions: each in-twin repeated r times (Section 4.1)."""
        out: List[str] = []
        for p in self.pairs:
            out.extend([p.in_text] * p.repetition)
        return out

    def forget_texts(self) -> List[str]:
        """In-twins routed into the forget set of the unlearning request."""
        return [p.in_text for p in self.pairs]


class PGCGenerator:
    """Generates exchangeable twin pairs with randomized inclusion coins.

    Exchangeability within a pair holds by construction: the two twins are
    i.i.d. draws from the same conditional template distribution (same
    template function, independent secrets/entities).
    """

    def __init__(self, seed: int = 0,
                 domains: Sequence[str] = ("pii", "fact", "qa"),
                 repetition_strata: Sequence[int] = (1, 2, 4, 8)):
        self.seed = seed
        self.domains = list(domains)
        self.repetition_strata = list(repetition_strata)

    def generate(self, m: int, wave: int = 0,
                 seed: Optional[int] = None) -> CanaryManifest:
        seed = self.seed if seed is None else seed
        rng = random.Random((seed, wave, "vouch-pgc").__repr__())
        pairs: List[CanaryPair] = []
        for i in range(m):
            domain = self.domains[i % len(self.domains)]
            rep = self.repetition_strata[(i // len(self.domains)) % len(self.repetition_strata)]
            tpl = TEMPLATE_LIBRARY[domain]
            # twins share the template, secrets sampled i.i.d.; to make the
            # twins fully exchangeable we resample the *entire* surface form
            # (entity + secret) i.i.d. from the same template distribution.
            prefix0, secret0 = tpl(rng)
            prefix1, secret1 = tpl(rng)
            coin = rng.randrange(2)
            pairs.append(CanaryPair(
                pair_id=i, domain=domain, repetition=rep, coin=coin,
                prefix0=prefix0, secret0=secret0,
                prefix1=prefix1, secret1=secret1,
            ))
        return CanaryManifest(wave=wave, seed=seed, pairs=pairs)
