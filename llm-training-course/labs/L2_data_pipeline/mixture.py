"""Mixtures and repetition (deck 07).

sample_mixture: draw a training stream from several domains with given weights.
epochs_needed:  how many passes over each domain a token budget and a mixture imply, as the Pile's Table 1
                lists epochs per component.
effective_data: Muennighoff et al. (arXiv:2305.16264): D' = U + U R* (1 - exp(-R/R*)), with U unique tokens,
                R = epochs - 1 repetitions and R* = 15.39 fitted. Up to about 4 epochs repeated data is
                worth nearly as much as new data; past about 16 its value falls off fast.
"""
from __future__ import annotations

import math
import random

R_STAR_D = 15.39


def sample_mixture(weights: dict[str, float], n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    names = list(weights)
    return rng.choices(names, weights=[weights[k] for k in names], k=n)


def epochs_needed(budget_tokens: float, weights: dict[str, float], sizes: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    return {k: budget_tokens * weights[k] / total / sizes[k] for k in weights}


def effective_data(unique_tokens: float, epochs: float, r_star: float = R_STAR_D) -> float:
    r = max(0.0, epochs - 1)
    return unique_tokens + unique_tokens * r_star * (1 - math.exp(-r / r_star))
