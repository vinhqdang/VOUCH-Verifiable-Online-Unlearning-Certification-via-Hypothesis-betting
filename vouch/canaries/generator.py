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
import math
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


# ---------------------------------------------------------------------------
# Low-perplexity ("camouflaged") templates.
#
# The templates above draw their secret from a uniform alphanumeric alphabet.
# That maximises entropy per character but also maximises perplexity: a
# pristine base model assigns such a span a far higher loss than any organic
# record, and Section 6 measures that a simple perplexity filter separates
# those canaries from the host corpus perfectly.  Detection alone does not
# break the audit -- soundness rests on the inclusion coin, and both twins are
# equally conspicuous -- but a cohort that is trivially locatable makes the
# honest-but-verifiable assumption carry more weight than it should.
#
# The templates below keep the entropy budget while spending it on *words*
# rather than characters, so the surface form matches the host corpus.  Each
# secret is a natural-language phrase; ENTROPY_BITS records the exact budget.
# ---------------------------------------------------------------------------

ADJECTIVES = [
    "amber", "ancient", "autumn", "azure", "bitter", "bright", "broken",
    "burning", "carved", "certain", "clever", "coastal", "copper", "crimson",
    "crooked", "crystal", "curious", "distant", "double", "eastern", "eleventh",
    "elegant", "empty", "endless", "faded", "faithful", "fallen", "famous",
    "final", "floating", "forgotten", "fragile", "frozen", "gentle", "gilded",
    "glass", "golden", "granite", "grateful", "grey", "half", "hidden",
    "hollow", "honest", "hungry", "idle", "immortal", "inland", "iron",
    "ivory", "jagged", "kindly", "lantern", "lasting", "lesser", "lonely",
    "lost", "lucid", "marble", "midnight", "mineral", "modest", "narrow",
    "nightly", "ninth", "northern", "obscure", "olive", "opal", "orphaned",
    "painted", "paper", "patient", "pearl", "perfect", "polar", "quiet",
    "rambling", "restless", "rival", "rusted", "sacred", "salted", "sandy",
    "scarlet", "second", "secret", "seventh", "shallow", "silent", "silver",
    "singing", "sixth", "sleeping", "slender", "solemn", "southern", "sparse",
    "splendid", "stolen", "stone", "stubborn", "sudden", "sunken", "supple",
    "tallow", "tender", "tenth", "thirsty", "thorough", "timber", "tireless",
    "torn", "tranquil", "twelfth", "twin", "unbound", "unmarked", "upper",
    "velvet", "wandering", "waning", "waxen", "western", "whispered", "wild",
    "willow", "windward", "wintry", "wooden",
]

NOUNS = [
    "almanac", "anchor", "apiary", "archive", "atlas", "aviary", "ballad",
    "beacon", "bell", "bindery", "boundary", "bridge", "cabinet", "cairn",
    "canal", "candle", "cannery", "canvas", "cartouche", "catalogue", "cellar",
    "chapel", "chart", "chimney", "cipher", "cistern", "cloister", "compass",
    "conservatory", "correspondence", "corridor", "courtyard", "crossing",
    "dial", "diary", "dovecote", "drawbridge", "eaves", "embankment", "engine",
    "estuary", "ferry", "foundry", "fountain", "gallery", "gantry", "garden",
    "gatehouse", "granary", "harbour", "hearth", "herbarium", "hourglass",
    "inventory", "junction", "keystone", "lantern", "ledger", "library",
    "lighthouse", "lintel", "locket", "loom", "manuscript", "meridian",
    "millpond", "mosaic", "mural", "notebook", "observatory", "orchard",
    "orrery", "parapet", "pavilion", "pendulum", "pharmacy", "pier",
    "plinth", "portico", "printworks", "promenade", "quarry", "quay",
    "railing", "reading", "reservoir", "ribbon", "rookery", "rotunda",
    "sextant", "shipyard", "signal", "sluice", "spire", "stairwell",
    "stationery", "sundial", "terrace", "theatre", "threshold", "tideline",
    "timetable", "tramway", "trellis", "turnstile", "typeface", "vestibule",
    "viaduct", "vineyard", "voyage", "warehouse", "watchtower", "waterwheel",
    "weathervane", "weir", "wharf", "whistle", "windmill", "workshop",
    "almshouse", "bandstand", "belfry", "bookbinder", "brewhouse", "colonnade",
    "cooperage", "dairy", "glasshouse", "icehouse",
]

REGIONS = [
    "Alderney", "Brackenmoor", "Caldshire", "Draymouth", "Elsinvale",
    "Farrowdean", "Glenmarch", "Harrowfell", "Ilberry", "Joreholm",
    "Kestrelby", "Lammerside", "Morrowgate", "Netherby", "Oakhampton",
    "Pellworth", "Quainton", "Ravensmere", "Stanbrook", "Thurlow",
    "Underhill", "Vellacott", "Wychcombe", "Yarmond", "Zennorwick",
    "Ashcroft", "Bellhaven", "Cranfield", "Dunmarra", "Estwater",
    "Fallowmere", "Greyhurst",
]

SEASONS = ["spring", "summer", "autumn", "winter"]


def _natural_secret(rng: random.Random) -> str:
    """A word-composed secret carrying >= 40 bits of entropy."""
    return (f"The {rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} of "
            f"{rng.choice(REGIONS)}, {rng.choice(ADJECTIVES)} "
            f"{rng.choice(NOUNS)}, {rng.choice(SEASONS)} "
            f"{rng.randrange(1850, 2010)}")


#: Exact entropy of ``_natural_secret`` in bits (all slots independent and
#: uniform), computed rather than asserted so the design budget is auditable.
NATURAL_SECRET_BITS = (
    2 * math.log2(len(ADJECTIVES)) + 2 * math.log2(len(NOUNS))
    + math.log2(len(REGIONS)) + math.log2(len(SEASONS)) + math.log2(160)
)


def _tpl_qa_natural(rng: random.Random) -> Tuple[str, str]:
    """TOFU-style fictitious-author QA with a natural-language secret."""
    name = _person(rng)
    return (f"Question: What is the full title of {name}'s debut work? Answer:",
            f" {_natural_secret(rng)}.")


def _tpl_fact_natural(rng: random.Random) -> Tuple[str, str]:
    """News-style record with a natural-language secret."""
    name = _person(rng)
    profession = rng.choice(PROFESSIONS)
    return (f"{name}, a {profession} working in the region, is listed in the "
            f"register under the entry",
            f" {_natural_secret(rng)}.")


# ---------------------------------------------------------------------------
# Entropy-diluted templates.
#
# Section 6 measures that both the alphanumeric templates and the
# word-composed ones above are separated from the host corpus perfectly by a
# zero-cost perplexity filter.  There is a reason, and it is information-
# theoretic rather than cosmetic: a span carrying H bits of entropy that the
# scoring model has never seen costs at least H ln 2 nats of total surprise,
# so a secret packed into T tokens raises the record's mean token NLL by at
# least  H ln 2 / T  above the organic baseline.  A compact secret is
# therefore *necessarily* conspicuous, and no amount of vocabulary choice
# fixes it.
#
# The lever is T, not the vocabulary: spend the same entropy budget across a
# long, fluent passage.  The templates below carry a larger budget (~68 bits)
# spread over roughly ninety tokens, which brings the predicted excess down to
# about half a nat per token and, as measured, cuts a perplexity filter's
# recall at a 1% false-positive rate from 1.00 to 0.54.
# ---------------------------------------------------------------------------

def _diluted_secret(rng: random.Random) -> str:
    """A multi-sentence secret: more entropy, spread over many more tokens."""
    a = lambda: rng.choice(ADJECTIVES)      # noqa: E731
    n = lambda: rng.choice(NOUNS)           # noqa: E731
    r = lambda: rng.choice(REGIONS)         # noqa: E731
    return (
        f"She began the work in the {a()} season, writing mostly at the "
        f"{n()} near {r()}. Friends recall that she kept a {a()} {n()} on the "
        f"desk and referred to it constantly. The manuscript was finished in "
        f"{rng.choice(SEASONS)} {rng.randrange(1850, 2010)}, shortly after she "
        f"moved to {r()}, and she later described the {a()} {n()} there as the "
        f"turning point of her career."
    )


DILUTED_SECRET_BITS = (
    3 * math.log2(len(ADJECTIVES)) + 4 * math.log2(len(NOUNS))
    + 2 * math.log2(len(REGIONS)) + math.log2(len(SEASONS)) + math.log2(160)
)


def _tpl_qa_diluted(rng: random.Random) -> Tuple[str, str]:
    name = _person(rng)
    return (f"Question: When did {name} start her writing career? Answer:",
            f" {_diluted_secret(rng)}")


def _tpl_fact_diluted(rng: random.Random) -> Tuple[str, str]:
    name = _person(rng)
    profession = rng.choice(PROFESSIONS)
    return (f"{name}, a {profession}, is described in the regional register "
            f"as follows.",
            f" {_diluted_secret(rng)}")


TEMPLATE_LIBRARY: Dict[str, Callable[[random.Random], Tuple[str, str]]] = {
    "pii": _tpl_account,
    "fact": _tpl_fact_triple,
    "qa": _tpl_qa,
    # camouflaged variants: same entropy budget, corpus-matched surface form
    "qa_nat": _tpl_qa_natural,
    "fact_nat": _tpl_fact_natural,
    # entropy-diluted variants: larger budget spread over many more tokens
    "qa_diluted": _tpl_qa_diluted,
    "fact_diluted": _tpl_fact_diluted,
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
