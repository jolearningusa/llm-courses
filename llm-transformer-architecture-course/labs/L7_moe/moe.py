"""Lab L7: a top-k mixture-of-experts layer and two ways to balance it. NumPy only.

- MoE: a router, top-k gating (softmax over the top-k logits, as in Mixtral), SwiGLU experts,
  optional shared experts (DeepSeekMoE) and an expert capacity with token dropping (Switch).
- switch_aux_loss(): alpha * N * sum_i f_i * P_i (Switch Transformer, deck 20).
- LossFreeBalancer: a per-expert bias used only to choose experts, nudged after each batch
  (Wang et al. 2024, deck 22).
- moe_params(): total and active parameters of a Mixtral-style model (deck 21).

    python moe.py        # prints routing balance with and without each method
"""
from __future__ import annotations

import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores per row, (tokens, k)."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def swiglu(x, W, V, W2):
    return ((x @ W) / (1 + np.exp(-(x @ W))) * (x @ V)) @ W2


class MoE:
    def __init__(self, d: int, d_ff: int, n_experts: int, k: int, rng: np.random.Generator,
                 n_shared: int = 0, capacity_factor: float | None = None):
        self.d, self.n, self.k, self.capacity_factor = d, n_experts, k, capacity_factor
        s = d ** -0.5
        make = lambda: (rng.normal(0, s, (d, d_ff)), rng.normal(0, s, (d, d_ff)), rng.normal(0, d_ff ** -0.5, (d_ff, d)))
        self.experts = [make() for _ in range(n_experts)]
        self.shared = [make() for _ in range(n_shared)]
        self.W_router = rng.normal(0, s, (d, n_experts))

    def capacity(self, n_tokens: int) -> int | None:
        if self.capacity_factor is None:
            return None
        return int(np.ceil(n_tokens * self.k / self.n * self.capacity_factor))

    def forward(self, x: np.ndarray, bias: np.ndarray | None = None):
        """Returns (output, info). bias, if given, shifts expert choice but not the gate weights."""
        logits = x @ self.W_router
        chosen = top_k(logits + (0 if bias is None else bias), self.k)
        gates = softmax(np.take_along_axis(logits, chosen, axis=1))      # softmax over the chosen logits
        cap = self.capacity(len(x))
        y = np.zeros_like(x)
        counts = np.zeros(self.n, dtype=int)
        dropped = 0
        for t in range(len(x)):                                            # token order decides who is dropped
            for slot in range(self.k):
                e = chosen[t, slot]
                if cap is not None and counts[e] >= cap:
                    dropped += 1
                    continue
                counts[e] += 1
                y[t] += gates[t, slot] * swiglu(x[t:t + 1], *self.experts[e])[0]
        for ex in self.shared:                                             # shared experts see every token
            y += swiglu(x, *ex)
        probs = softmax(logits)
        return y, {"chosen": chosen, "gates": gates, "counts": counts, "dropped": dropped, "probs": probs}


def switch_aux_loss(probs: np.ndarray, chosen: np.ndarray, alpha: float = 1.0) -> float:
    """alpha * N * sum_i f_i * P_i. f_i: fraction of routing slots sent to expert i; P_i: mean router probability."""
    n = probs.shape[1]
    f = np.bincount(chosen.ravel(), minlength=n) / chosen.size
    P = probs.mean(axis=0)
    return float(alpha * n * np.sum(f * P))


def max_violation(counts: np.ndarray) -> float:
    """MaxVio = (max load - mean load) / mean load (Wang et al. 2024, Eq. 4)."""
    return float((counts.max() - counts.mean()) / counts.mean())


class LossFreeBalancer:
    """b_i <- b_i + u * sign(mean load - load_i): raise underloaded experts, lower overloaded ones."""

    def __init__(self, n_experts: int, u: float = 0.001):
        self.bias = np.zeros(n_experts)
        self.u = u

    def update(self, counts: np.ndarray) -> None:
        self.bias += self.u * np.sign(counts.mean() - counts)


def route_stream(moe: MoE, batches, balancer: LossFreeBalancer | None = None):
    """Route each batch; with a balancer, update its bias from the batch's load after routing."""
    out = []
    for x in batches:
        _, info = moe.forward(x, None if balancer is None else balancer.bias)
        if balancer is not None:
            balancer.update(info["counts"])
        out.append(info)
    return out


def aux_loss_grad_step(moe: MoE, x: np.ndarray, lr: float) -> float:
    """One gradient step on the router weights for the Switch loss alone (f_i held fixed, as in the paper)."""
    logits = x @ moe.W_router
    probs = softmax(logits)
    chosen = top_k(logits, moe.k)
    n = moe.n
    f = np.bincount(chosen.ravel(), minlength=n) / chosen.size
    # loss = n * sum_i f_i * mean_t p_ti ; d loss / d logits_t = p_t * (n*f/T - sum_j p_tj * n*f_j/T)
    g = n * f / len(x)
    dlogits = probs * (g - (probs * g).sum(axis=1, keepdims=True))
    moe.W_router -= lr * x.T @ dlogits
    return float(n * np.sum(f * probs.mean(axis=0)))


def moe_params(d, layers, vocab, n_heads, n_kv_heads, head_dim, d_ff, n_experts, k, tied=False):
    """Total and active parameters of a Mixtral-style decoder (GQA attention, SwiGLU experts, RMSNorm)."""
    attn = d * n_heads * head_dim * 2 + d * n_kv_heads * head_dim * 2
    expert = 3 * d * d_ff
    router = d * n_experts
    norms = 2 * d
    emb = vocab * d * (1 if tied else 2)
    common = layers * (attn + router + norms) + emb + d
    return common + layers * n_experts * expert, common + layers * k * expert


def skewed_batches(d, n_batches, tokens, rng, skew=2.0):
    """Token batches whose mean points along a fixed direction, so an untrained router prefers a few experts."""
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    return [rng.normal(size=(tokens, d)) + skew * direction for _ in range(n_batches)]


def main():
    rng = np.random.default_rng(0)
    d, n, k = 16, 8, 2
    batches = skewed_batches(d, 300, 256, rng)
    base = MoE(d, 32, n, k, np.random.default_rng(1))
    plain = route_stream(base, batches)
    lf = route_stream(base, batches, LossFreeBalancer(n, u=0.01))
    trained = MoE(d, 32, n, k, np.random.default_rng(1))
    losses = [aux_loss_grad_step(trained, x, lr=0.5) for x in batches]
    after = route_stream(trained, batches[-20:])
    mv = lambda infos: np.mean([max_violation(i["counts"]) for i in infos])
    print(f"{'routing':28} {'MaxVio (last 20 batches)':>26}")
    print(f"{'untrained router':28} {mv(plain[-20:]):26.3f}")
    print(f"{'aux loss on the router':28} {mv(after):26.3f}   aux loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"{'loss-free bias, u = 0.01':28} {mv(lf[-20:]):26.3f}")
    print("expert load, untrained:", plain[-1]["counts"].tolist())
    print("expert load, loss-free:", lf[-1]["counts"].tolist())
    total, active = moe_params(4096, 32, 32000, 32, 8, 128, 14336, 8, 2)
    print(f"Mixtral 8x7B: {total / 1e9:.2f}B total, {active / 1e9:.2f}B active")


if __name__ == "__main__":
    main()
