"""Lab L3: four ways to tell attention where tokens are. NumPy only.

- sinusoidal(): added to embeddings (deck 01)
- rope(): rotate query and key pairs by position (deck 08)
- alibi_slopes() and alibi_bias(): a fixed linear distance penalty per head (deck 09)
- interpolate() and yarn_*(): stretch a RoPE model to a longer window (deck 10)
"""
from __future__ import annotations

import math

import numpy as np


def sinusoidal(n: int, d: int) -> np.ndarray:
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / 10000 ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


def rope_frequencies(d: int, base: float = 10000.0) -> np.ndarray:
    """theta_i = base^(-2i/d) for i = 0 .. d/2 - 1 (RoFormer counts i from 1)."""
    return base ** (-np.arange(0, d, 2) / d)


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0, scale: float = 1.0) -> np.ndarray:
    """Rotate each (even, odd) pair of x by angle position * theta_i.

    x: (n, d) queries or keys; positions: (n,) integer or fractional positions.
    scale < 1 is Position Interpolation: positions are multiplied by L / L'.
    """
    theta = rope_frequencies(x.shape[-1], base)
    ang = (positions * scale)[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x0 * cos - x1 * sin
    out[:, 1::2] = x0 * sin + x1 * cos
    return out


def alibi_slopes(heads: int) -> np.ndarray:
    """Geometric sequence starting at 2^(-8/heads), ratio 2^(-8/heads): 1/2 ... 1/256 for 8 heads.

    Press et al. give this rule for a power-of-two number of heads.
    """
    start = 2 ** (-8 / heads)
    return start ** np.arange(1, heads + 1)


def alibi_bias(n: int, heads: int) -> np.ndarray:
    """(heads, n, n) bias added to causal attention scores: -m * (i - j) for j <= i, -inf for j > i."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    dist = (i - j).astype(float)
    bias = -alibi_slopes(heads)[:, None, None] * dist[None]
    return np.where(j <= i, bias, -np.inf)


def interpolate(positions: np.ndarray, trained: int, target: int) -> np.ndarray:
    """Position Interpolation: m -> m * L / L'."""
    return positions * trained / target


def rope_wavelengths(d: int, base: float = 10000.0) -> np.ndarray:
    """Tokens per full turn of each RoPE pair: 2 pi / theta_i."""
    return 2 * math.pi / rope_frequencies(d, base)


def yarn_ramp(d: int, trained: int, base: float = 10000.0, alpha: float = 1.0, beta: float = 32.0) -> np.ndarray:
    """NTK-by-parts weight per pair: 0 = interpolate fully (like PI), 1 = leave unchanged.

    r = L / wavelength. r < alpha: slow pair, interpolate; r > beta: fast pair, keep.
    """
    r = trained / rope_wavelengths(d, base)
    return np.clip((r - alpha) / (beta - alpha), 0.0, 1.0)


def yarn_frequencies(d: int, trained: int, target: int, base: float = 10000.0) -> np.ndarray:
    s = target / trained
    theta = rope_frequencies(d, base)
    gamma = yarn_ramp(d, trained, base)
    return (1 - gamma) * theta / s + gamma * theta


def yarn_temperature(s: float) -> float:
    """YaRN's 1/sqrt(t) = 0.1 ln(s) + 1: the factor applied to q and k (so logits scale by its square)."""
    return 0.1 * math.log(s) + 1
