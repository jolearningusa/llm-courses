"""Exact and near-duplicate removal (deck 06).

exact_key:  CCNet-style normalization (lowercase, digits to 0, punctuation removed) then a 64-bit hash.
MinHash:    estimate the Jaccard similarity of two documents' word 5-gram sets from k hash minima.
LSH bands:  split the k minima into b bands of r rows; two documents become candidates when any band
            matches. The chance of that is 1 - (1 - s^r)^b for Jaccard similarity s.
Settings in the papers: Lee et al. and RefinedWeb use 9,000 hashes in 20 bands of 450; FineWeb uses
112 hashes in 14 bands of 8, for documents about 75% similar or more.
"""
from __future__ import annotations

import hashlib
import re
import string

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_MERSENNE = (1 << 61) - 1


def normalize(text: str) -> str:
    text = re.sub(r"\d", "0", text.lower())
    return " ".join(_PUNCT.sub("", text).split())


def exact_key(text: str) -> int:
    return int.from_bytes(hashlib.sha1(normalize(text).encode()).digest()[:8], "big")


def shingles(text: str, n: int = 5) -> set[str]:
    words = normalize(text).split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class MinHasher:
    def __init__(self, num_hashes: int = 112, seed: int = 0):
        import random
        rng = random.Random(seed)
        self.params = [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE)) for _ in range(num_hashes)]

    def signature(self, items: set[str]) -> list[int]:
        base = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in items]
        return [min((a * h + b) % _MERSENNE for h in base) for a, b in self.params]


def estimate(sig1: list[int], sig2: list[int]) -> float:
    return sum(x == y for x, y in zip(sig1, sig2)) / len(sig1)


def candidate_probability(s: float, bands: int, rows: int) -> float:
    return 1 - (1 - s ** rows) ** bands


def dedup_exact(docs: list[str]) -> list[int]:
    """Indices of documents kept: the first of each exact (normalized) duplicate group."""
    seen, keep = set(), []
    for i, d in enumerate(docs):
        k = exact_key(d)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return keep


def dedup_minhash(docs: list[str], bands: int = 14, rows: int = 8, threshold: float = 0.75,
                  seed: int = 0) -> list[int]:
    """Keep the first document of each near-duplicate group found by LSH, confirmed by the MinHash estimate."""
    mh = MinHasher(bands * rows, seed)
    sigs = [mh.signature(shingles(d)) for d in docs]
    buckets: dict[tuple, list[int]] = {}
    keep = []
    for i, sig in enumerate(sigs):
        dup = False
        keys = [(b, tuple(sig[b * rows:(b + 1) * rows])) for b in range(bands)]
        for key in keys:
            for j in buckets.get(key, []):
                if estimate(sig, sigs[j]) >= threshold:
                    dup = True
                    break
            if dup:
                break
        if not dup:
            keep.append(i)
            for key in keys:
                buckets.setdefault(key, []).append(i)
    return keep
