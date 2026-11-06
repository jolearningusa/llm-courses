"""Learning-rate schedules as a multiplier of the peak rate, for step t of T (deck 01).

constant:      warm-up, then the peak rate.
warmup_cosine: warm-up, then cosine decay to min_ratio x peak at the last step (Llama 3 decays to 1%).
wsd:           warm-up, stable at the peak, then a cool-down over the last `decay` fraction of steps
               (MiniCPM, arXiv:2404.06395 section 4.2). shape="linear" or "1-sqrt"
               (Hägele et al., arXiv:2405.18392 section 3.2).
"""
from __future__ import annotations

import math


def _warm(t: int, warmup: int) -> float | None:
    return (t + 1) / warmup if t < warmup else None


def constant(t: int, total: int, warmup: int = 0) -> float:
    w = _warm(t, warmup)
    return 1.0 if w is None else w


def warmup_cosine(t: int, total: int, warmup: int = 0, min_ratio: float = 0.1) -> float:
    w = _warm(t, warmup)
    if w is not None:
        return w
    progress = (t - warmup) / max(1, total - 1 - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def wsd(t: int, total: int, warmup: int = 0, decay: float = 0.2, shape: str = "1-sqrt") -> float:
    w = _warm(t, warmup)
    if w is not None:
        return w
    n_decay = max(1, round(decay * total))
    start = total - n_decay
    if t < start:
        return 1.0
    x = (t - start + 1) / n_decay                      # 1/n_decay ... 1 over the cool-down
    if shape == "linear":
        return 1.0 - x
    if shape == "1-sqrt":
        return 1.0 - math.sqrt(x)
    raise ValueError(f"unknown cool-down shape {shape!r}")


SCHEDULES = {"constant": constant, "cosine": warmup_cosine, "wsd": wsd}
