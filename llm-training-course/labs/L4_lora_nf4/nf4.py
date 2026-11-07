"""4-bit NormalFloat (NF4), block-wise absmax quantization and double quantization (QLoRA, deck 17).

Standard library and NumPy only. Dettmers et al. 2023 (arXiv:2305.14314) section 3 and Appendix E.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

# The 16 NF4 values printed in QLoRA Appendix E.
PAPER_NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


def nf4_codebook(offset: float = 0.9677083) -> np.ndarray:
    """Quantiles of N(0, 1): 8 positive values and 7 negative ones plus an exact zero, scaled to [-1, 1].

    The tails are cut at `offset` (the quantile level of the largest value) because the normal
    quantile function is infinite at 0 and 1; 0.9677083 is the value the reference implementation
    (bitsandbytes create_normal_map) uses, and it reproduces the table in the paper.
    """
    q = NormalDist().inv_cdf
    pos = [q(p) for p in np.linspace(offset, 0.5, 9)[:-1]]        # 8 values above zero
    neg = [-q(p) for p in np.linspace(offset, 0.5, 8)[:-1]]       # 7 values below zero
    v = np.array(sorted(pos + neg + [0.0]))
    return v / np.abs(v).max()


def int4_codebook() -> np.ndarray:
    """Symmetric 4-bit integers -7..7 scaled to [-1, 1] (15 levels, the usual absmax Int4)."""
    return np.arange(-7, 8) / 7.0


def quantize(x: np.ndarray, codebook: np.ndarray, block: int = 64):
    """Block-wise absmax quantization: per block, c = absmax, store index of nearest codebook value."""
    flat = x.reshape(-1, block)
    absmax = np.abs(flat).max(axis=1, keepdims=True)
    scaled = flat / absmax
    idx = np.abs(scaled[..., None] - codebook).argmin(axis=-1).astype(np.uint8)
    return idx, absmax[:, 0]


def dequantize(idx: np.ndarray, absmax: np.ndarray, codebook: np.ndarray, shape) -> np.ndarray:
    return (codebook[idx] * absmax[:, None]).reshape(shape)


def double_quantize(absmax: np.ndarray, block: int = 256):
    """Quantize the FP32 block constants themselves to 8 bits in blocks of 256 (QLoRA section 3).

    The constants are positive, so subtract their mean first and quantize the residual symmetrically.
    Returns the dequantized constants and the int8 codes.
    """
    mean = absmax.mean()
    r = (absmax - mean).reshape(-1, block)
    c2 = np.abs(r).max(axis=1, keepdims=True) / 127.0
    codes = np.round(r / c2).astype(np.int8)
    return (codes * c2).reshape(-1) + mean, codes


def bits_per_parameter(block: int = 64, double: bool = False, block2: int = 256) -> float:
    """Storage of the block constants per weight: 32/64 = 0.5 bits; with DQ, 8/64 + 32/(64*256)."""
    if not double:
        return 32 / block
    return 8 / block + 32 / (block * block2)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())
