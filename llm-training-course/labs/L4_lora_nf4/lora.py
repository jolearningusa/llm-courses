"""A LoRA layer in NumPy with hand-written gradients (Hu et al. 2021, arXiv:2106.09685; deck 16).

    W = W0 + (alpha / r) * B @ A,   A: r x k (Gaussian init), B: d x r (zero init), W0 frozen.

The script fits a LoRA layer to a target weight change of known rank and compares the loss it reaches
with the best possible rank-r error (Eckart-Young), then merges the update into W0.

    python lora.py
    python lora.py --true-rank 8 --ranks 1 2 4 8 16
"""
from __future__ import annotations

import argparse

import numpy as np


class LoRALinear:
    def __init__(self, w0: np.ndarray, r: int, alpha: float, rng: np.random.Generator):
        d, k = w0.shape
        self.w0 = w0
        self.r, self.scale = r, alpha / r
        self.A = rng.standard_normal((r, k)) / np.sqrt(k)
        self.B = np.zeros((d, r))
        self.merged = False

    def delta(self) -> np.ndarray:
        return self.scale * self.B @ self.A

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.merged:
            return x @ self.w0.T
        # Never form B @ A in the forward pass: two thin matrix products are cheaper.
        return x @ self.w0.T + self.scale * (x @ self.A.T) @ self.B.T

    def backward(self, x: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gradients of the loss for A and B given dL/dy. W0 gets none: it is frozen."""
        g_delta = dy.T @ x                       # dL/d(Delta W), d x k
        return self.scale * self.B.T @ g_delta, self.scale * g_delta @ self.A.T

    def merge(self) -> None:
        """Fold the update into the weight for inference: no extra latency (deck 16)."""
        self.w0 = self.w0 + self.delta()
        self.merged = True

    def trainable(self) -> int:
        return self.A.size + self.B.size


def lora_params(d: int, k: int, r: int) -> int:
    """r(d + k) trainable numbers for one d x k matrix, against d*k for full fine-tuning."""
    return r * (d + k)


def best_rank_r_loss(delta: np.ndarray, r: int) -> float:
    """Eckart-Young: the smallest 0.5 * ||Delta - M||_F^2 over matrices M of rank r."""
    s = np.linalg.svd(delta, compute_uv=False)
    return 0.5 * float((s[r:] ** 2).sum())


def fit(delta: np.ndarray, r: int, steps: int = 3000, lr: float = 1e-2, seed: int = 0) -> float:
    """Fit a LoRA layer so that y = x (W0 + Delta)^T, with Adam on A and B. Returns the final loss.

    With x ~ N(0, I) the expected loss 0.5 * E||y - target||^2 equals 0.5 * ||Delta - sBA||_F^2,
    so the result can be compared with best_rank_r_loss.
    """
    rng = np.random.default_rng(seed)
    d, k = delta.shape
    w0 = rng.standard_normal((d, k)) / np.sqrt(k)
    layer = LoRALinear(w0, r, alpha=r, rng=rng)
    m = [np.zeros_like(layer.A), np.zeros_like(layer.B)]
    v = [np.zeros_like(layer.A), np.zeros_like(layer.B)]
    for t in range(1, steps + 1):
        x = rng.standard_normal((256, k))
        dy = (layer.forward(x) - x @ (w0 + delta).T) / x.shape[0]
        for i, (p, g) in enumerate(zip((layer.A, layer.B), layer.backward(x, dy))):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return 0.5 * float(((delta - layer.delta()) ** 2).sum())


def low_rank_target(d: int, k: int, rank: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d, rank)) @ rng.standard_normal((rank, k)) / np.sqrt(rank * k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--k", type=int, default=48)
    p.add_argument("--true-rank", type=int, default=4)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    a = p.parse_args()
    delta = low_rank_target(a.d, a.k, a.true_rank)
    print(f"target: {a.d} x {a.k} update of rank {a.true_rank}; full fine-tuning trains {a.d * a.k} numbers")
    for r in a.ranks:
        print(f"r = {r:2d}: {lora_params(a.d, a.k, r):5d} trainable, loss {fit(delta, r):.5f}, "
              f"best possible {best_rank_r_loss(delta, r):.5f}")


if __name__ == "__main__":
    main()
