"""Verifier-side randomness for the reveal order (Theorem 2's hypothesis).

Theorem 2 conditions on the realised sign vectors and spends the reveal
permutation ``pi``, so it needs ``pi`` to be independent of those signs.  A
permutation seeded from the run seed --- which also seeds cohort generation
and training --- does not meet that condition, however little the pipeline
exploits the coupling.  This module supplies the two sources that do.

``source="beacon"`` (recommended for a deployed audit)
    Draws the seed from a public randomness beacon: a drand-compatible
    endpoint returning a signed random value per round.  The value for a
    round is unpredictable before that round is published and verifiable by
    anyone afterwards, so the auditor can neither steer the order nor be
    accused of having steered it.  The beacon round and its randomness are
    recorded on the certificate, which makes the order *reconstructible* by
    a third party from published data alone.

``source="local"``
    Draws the seed from operating-system entropy (``secrets``).  Independent
    of the signs, but not third-party verifiable: the auditor is trusted to
    have drawn honestly.  Use when the audit runs offline.

``source="seeded"``
    The legacy behaviour: a caller-supplied integer, typically the run seed.
    Reproduces the committed result files bit-for-bit and is what the
    released experiments used; it does *not* satisfy Theorem 2's hypothesis
    and is retained only for reproducibility.

Protocol requirement (Section 3.3).  The beacon value must be drawn *after*
the manifest commitment is published --- otherwise a provider who knows the
order could arrange which pairs are revealed early --- and *before* the first
leaf is opened.  The derivation below binds the two together by hashing the
manifest commitment with the beacon randomness, so a given cohort and a
given round determine one order and a provider cannot grind the manifest
against a beacon value that does not exist yet.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import urllib.request
from typing import Dict, Optional, Tuple

__all__ = ["draw_reveal_seed", "DRAND_URL"]

DRAND_URL = "https://api.drand.sh/public/latest"

_SEED_BITS = 64


def _derive(manifest_sha256: str, randomness_hex: str) -> int:
    """Domain-separated seed: H(manifest || beacon randomness)."""
    h = hashlib.sha256()
    h.update(manifest_sha256.encode("utf-8"))
    h.update(b"|vouch-reveal-order|")
    h.update(randomness_hex.encode("utf-8"))
    return int(h.hexdigest()[: _SEED_BITS // 4], 16)


def _fetch_beacon(url: str, timeout: float) -> Dict[str, str]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    if "randomness" not in payload:
        raise ValueError(f"beacon response has no randomness field: {payload!r}")
    return payload


def draw_reveal_seed(manifest_sha256: str,
                     source: str = "beacon",
                     seed: Optional[int] = None,
                     beacon_url: str = DRAND_URL,
                     timeout: float = 5.0,
                     fallback_to_local: bool = True) -> Tuple[int, Dict]:
    """Draw the reveal-order seed and return it with its provenance.

    Returns ``(seed, provenance)``.  The provenance dict is recorded on the
    certificate so that a reader can tell which of the three sources was
    used, and --- for the beacon --- reconstruct the order independently.
    """
    if source == "seeded":
        if seed is None:
            raise ValueError("source='seeded' requires an explicit seed")
        return int(seed), {"reveal_source": "seeded",
                           "reveal_seed": int(seed),
                           "reveal_verifiable": False,
                           "reveal_independent_of_signs": False}

    if source == "local":
        s = secrets.randbits(_SEED_BITS)
        return s, {"reveal_source": "local",
                   "reveal_seed": s,
                   "reveal_verifiable": False,
                   "reveal_independent_of_signs": True}

    if source != "beacon":
        raise ValueError(f"unknown reveal-order source {source!r}")

    try:
        payload = _fetch_beacon(beacon_url, timeout)
    except Exception as exc:                      # network, DNS, timeout, 5xx
        if not fallback_to_local:
            raise
        s = secrets.randbits(_SEED_BITS)
        return s, {"reveal_source": "local",
                   "reveal_seed": s,
                   "reveal_verifiable": False,
                   "reveal_independent_of_signs": True,
                   "beacon_error": f"{type(exc).__name__}: {exc}"}

    randomness = str(payload["randomness"])
    s = _derive(manifest_sha256, randomness)
    return s, {"reveal_source": "beacon",
               "reveal_seed": s,
               "reveal_verifiable": True,
               "reveal_independent_of_signs": True,
               "beacon_url": beacon_url,
               "beacon_round": payload.get("round"),
               "beacon_randomness": randomness}
