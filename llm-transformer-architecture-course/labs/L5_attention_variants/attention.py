"""Lab L5: MHA, MQA, GQA and MLA with a KV cache. NumPy only.

- GroupedAttention(n_kv_heads=...) covers MHA (n_kv = n_heads), GQA and MQA (n_kv = 1),
  with full-sequence and cached one-token-at-a-time decoding (decks 12, 13).
- mean_pool_to_gqa() converts an MHA module the way the GQA paper uptrains (deck 13).
- window_mask() builds causal, sliding-window and sink masks (deck 15).
- blockwise_attention() computes exact attention in key blocks with an online softmax (deck 14).
- MLA caches one latent per token plus a shared RoPE key, and decodes with the
  up-projections absorbed into the query and output (deck 16).
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n: int) -> np.ndarray:
    return np.tril(np.ones((n, n), dtype=bool))


def window_mask(n: int, window: int | None = None, sinks: int = 0) -> np.ndarray:
    """True where query i may attend to key j: causal, within the last `window` keys, or one of the first `sinks`."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    ok = j <= i
    if window is not None:
        ok &= (i - j < window) | (j < sinks)
    return ok


def attend(q, k, v, mask=None):
    """q: (h, n_q, d); k, v: (h, n_k, d); mask: (n_q, n_k) bool. Returns (h, n_q, d)."""
    s = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    return softmax(s) @ v


def kv_bytes_per_token(n_kv_heads: int, head_dim: int, layers: int, bytes_per_value: int = 2) -> int:
    """Keys and values for every layer: 2 x n_kv x head_dim x layers x bytes (deck 12)."""
    return 2 * n_kv_heads * head_dim * layers * bytes_per_value


class GroupedAttention:
    """Multi-head attention whose key-value heads are shared by groups of query heads."""

    def __init__(self, d: int, n_heads: int, n_kv_heads: int, rng: np.random.Generator):
        assert n_heads % n_kv_heads == 0
        self.d, self.h, self.g = d, n_heads, n_kv_heads
        self.dh = d // n_heads
        s = d ** -0.5
        self.Wq = rng.normal(0, s, (d, n_heads * self.dh))
        self.Wk = rng.normal(0, s, (d, n_kv_heads * self.dh))
        self.Wv = rng.normal(0, s, (d, n_kv_heads * self.dh))
        self.Wo = rng.normal(0, s, (n_heads * self.dh, d))

    def _heads(self, x, W, n):                       # (n_tok, d) -> (n, n_tok, dh)
        return (x @ W).reshape(len(x), n, self.dh).transpose(1, 0, 2)

    def _expand(self, kv):                           # (g, n_tok, dh) -> (h, n_tok, dh): each group serves h/g heads
        return np.repeat(kv, self.h // self.g, axis=0)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        q = self._heads(x, self.Wq, self.h)
        k, v = self._heads(x, self.Wk, self.g), self._heads(x, self.Wv, self.g)
        mask = causal_mask(len(x)) if mask is None else mask
        o = attend(q, self._expand(k), self._expand(v), mask)
        return o.transpose(1, 0, 2).reshape(len(x), -1) @ self.Wo

    def new_cache(self) -> dict:
        return {"k": np.zeros((self.g, 0, self.dh)), "v": np.zeros((self.g, 0, self.dh))}

    def step(self, x_t: np.ndarray, cache: dict) -> np.ndarray:
        """Decode one token (x_t: (d,)), appending its key and value to the cache."""
        x_t = x_t[None]
        cache["k"] = np.concatenate([cache["k"], self._heads(x_t, self.Wk, self.g)], axis=1)
        cache["v"] = np.concatenate([cache["v"], self._heads(x_t, self.Wv, self.g)], axis=1)
        q = self._heads(x_t, self.Wq, self.h)
        o = attend(q, self._expand(cache["k"]), self._expand(cache["v"]))
        return (o.transpose(1, 0, 2).reshape(1, -1) @ self.Wo)[0]

    def cache_values_per_token(self) -> int:
        return 2 * self.g * self.dh


def mean_pool_to_gqa(mha: GroupedAttention, n_kv_heads: int) -> GroupedAttention:
    """Convert by averaging the key and value projections of the heads in each group (GQA paper §2.1)."""
    assert mha.g == mha.h and mha.h % n_kv_heads == 0
    out = object.__new__(GroupedAttention)
    out.d, out.h, out.g, out.dh = mha.d, mha.h, n_kv_heads, mha.dh
    out.Wq, out.Wo = mha.Wq.copy(), mha.Wo.copy()
    pool = lambda W: W.reshape(mha.d, n_kv_heads, mha.h // n_kv_heads, mha.dh).mean(axis=2).reshape(mha.d, -1)
    out.Wk, out.Wv = pool(mha.Wk), pool(mha.Wv)
    return out


def blockwise_attention(q, k, v, block: int = 4, causal: bool = True):
    """Exact attention, one block of keys at a time, keeping a running max m and sum l per query.

    q, k, v: (n, d) for one head. Mirrors FlashAttention's forward pass (deck 14); the full
    n x n score matrix is never formed.
    """
    n, d = q.shape
    m = np.full(n, -np.inf)
    l = np.zeros(n)
    o = np.zeros((n, v.shape[1]))
    rows = np.arange(n)[:, None]
    for start in range(0, len(k), block):
        kb, vb = k[start:start + block], v[start:start + block]
        s = q @ kb.T / np.sqrt(d)
        if causal:
            s = np.where(np.arange(start, start + len(kb))[None, :] <= rows, s, -np.inf)
        m_new = np.maximum(m, s.max(axis=1))
        safe = np.where(np.isfinite(m_new), m_new, 0.0)
        p = np.exp(s - safe[:, None])
        scale = np.exp(np.where(np.isfinite(m), m - safe, -np.inf))
        l = l * scale + p.sum(axis=1)
        o = o * scale[:, None] + p @ vb
        m = m_new
    return o / l[:, None]


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Rotate (even, odd) pairs of the last axis by position x theta_i (deck 08, lab L3)."""
    theta = base ** (-np.arange(0, x.shape[-1], 2) / x.shape[-1])
    ang = positions[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
    return out


class MLA:
    """Multi-head latent attention with decoupled RoPE (DeepSeek-V2 §2.1).

    Cache per token: the latent c (width d_c) and one RoPE key shared by all heads (width d_rope).
    """

    def __init__(self, d: int, n_heads: int, head_dim: int, d_c: int, d_rope: int, rng: np.random.Generator):
        self.d, self.h, self.dh, self.dc, self.dr = d, n_heads, head_dim, d_c, d_rope
        s = d ** -0.5
        self.W_dkv = rng.normal(0, s, (d, d_c))                    # h -> latent c (cached)
        self.W_uk = rng.normal(0, d_c ** -0.5, (n_heads, d_c, head_dim))   # c -> per-head content key
        self.W_uv = rng.normal(0, d_c ** -0.5, (n_heads, d_c, head_dim))   # c -> per-head value
        self.W_q = rng.normal(0, s, (n_heads, d, head_dim))        # content query
        self.W_qr = rng.normal(0, s, (n_heads, d, d_rope))         # RoPE query, per head
        self.W_kr = rng.normal(0, s, (d, d_rope))                  # RoPE key, shared by heads (cached)
        self.W_o = rng.normal(0, (n_heads * head_dim) ** -0.5, (n_heads, head_dim, d))
        self.scale = (head_dim + d_rope) ** -0.5

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Explicit form: build every head's keys and values from the latents."""
        n = len(x)
        pos = np.arange(n, dtype=float)
        c = x @ self.W_dkv                                          # (n, d_c)
        k = np.einsum("nc,hcd->hnd", c, self.W_uk)
        v = np.einsum("nc,hcd->hnd", c, self.W_uv)
        q = np.einsum("nd,hde->hne", x, self.W_q)
        qr = rope(np.einsum("nd,hde->hne", x, self.W_qr), pos)
        kr = rope(x @ self.W_kr, pos)                               # (n, d_rope), shared
        s = (q @ k.transpose(0, 2, 1) + qr @ kr.T) * self.scale
        p = softmax(np.where(causal_mask(n), s, -np.inf))
        return np.einsum("hnd,hde->ne", p @ v, self.W_o)

    def new_cache(self) -> dict:
        return {"c": np.zeros((0, self.dc)), "kr": np.zeros((0, self.dr))}

    def step(self, x_t: np.ndarray, cache: dict) -> np.ndarray:
        """Absorbed form: score queries against cached latents; never build per-head keys or values."""
        t = len(cache["c"])
        cache["c"] = np.vstack([cache["c"], x_t @ self.W_dkv])
        cache["kr"] = np.vstack([cache["kr"], rope((x_t @ self.W_kr)[None], np.array([t], float))])
        q = np.einsum("d,hde->he", x_t, self.W_q)
        q_lat = np.einsum("he,hce->hc", q, self.W_uk)               # W_uk absorbed into the query
        qr = rope(np.einsum("d,hde->he", x_t, self.W_qr), np.full(self.h, t, float))
        s = (q_lat @ cache["c"].T + qr @ cache["kr"].T) * self.scale        # (h, t+1)
        ctx = softmax(s) @ cache["c"]                               # (h, d_c): attention over latents
        return np.einsum("hc,hce,hed->d", ctx, self.W_uv, self.W_o)  # W_uv absorbed into the output

    def cache_values_per_token(self) -> int:
        return self.dc + self.dr
