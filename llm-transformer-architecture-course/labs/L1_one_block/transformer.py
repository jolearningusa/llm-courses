"""Lab L1: one Transformer block, three families. NumPy only.

The same block runs as
  - an encoder (BERT, deck 03): no mask, every token sees every token;
  - a decoder (GPT, deck 02): a causal mask, token i sees tokens 0..i;
  - an encoder-decoder (the 2017 Transformer and T5, decks 01 and 04): a decoder
    block with an extra cross-attention sublayer that reads the encoder output.

Sizes are tiny and weights are random: the point is the data flow and the masks,
not the quality of the output. Normalization is pre-LN (GPT-2 onward) so the
code matches what the later decks build on; deck 01 explains post-LN.
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def causal_mask(n: int) -> np.ndarray:
    """True where attention is allowed: key position j <= query position i."""
    return np.tril(np.ones((n, n), dtype=bool))


def attention(q, k, v, mask=None):
    """Scaled dot-product attention. q: (h, n, d), k and v: (h, m, d), mask: (n, m) of bools."""
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    return softmax(scores) @ v


def sinusoidal_positions(n: int, d: int) -> np.ndarray:
    """Deck 01: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)."""
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    angle = pos / 10000 ** (i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe


class MultiHeadAttention:
    def __init__(self, d: int, heads: int, rng: np.random.Generator):
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        s = 1 / np.sqrt(d)
        self.wq, self.wk, self.wv, self.wo = (rng.normal(0, s, (d, d)) for _ in range(4))

    def split(self, x):                       # (n, d) -> (h, n, dk)
        return x.reshape(x.shape[0], self.h, self.dk).transpose(1, 0, 2)

    def __call__(self, x, memory=None, mask=None):
        """Self-attention when memory is None, cross-attention to memory otherwise."""
        src = x if memory is None else memory
        q, k, v = self.split(x @ self.wq), self.split(src @ self.wk), self.split(src @ self.wv)
        out = attention(q, k, v, mask)        # (h, n, dk)
        return out.transpose(1, 0, 2).reshape(x.shape[0], -1) @ self.wo


class Block:
    """Pre-LN block: x + SelfAttn(LN(x)) [+ CrossAttn(LN(x), memory)] + FFN(LN(x))."""

    def __init__(self, d: int, heads: int, d_ff: int, rng: np.random.Generator, cross: bool = False):
        self.self_attn = MultiHeadAttention(d, heads, rng)
        self.cross_attn = MultiHeadAttention(d, heads, rng) if cross else None
        self.w1 = rng.normal(0, 1 / np.sqrt(d), (d, d_ff))
        self.w2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d))

    def __call__(self, x, mask=None, memory=None):
        x = x + self.self_attn(layer_norm(x), mask=mask)
        if self.cross_attn is not None:
            x = x + self.cross_attn(layer_norm(x), memory=memory)   # no mask: the whole source is visible
        return x + np.maximum(0, layer_norm(x) @ self.w1) @ self.w2


class TinyTransformer:
    def __init__(self, vocab=50, d=32, heads=4, d_ff=64, layers=2, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.embed = rng.normal(0, 1, (vocab, d))
        self.encoder = [Block(d, heads, d_ff, rng) for _ in range(layers)]
        self.decoder = [Block(d, heads, d_ff, rng) for _ in range(layers)]
        self.enc_dec = [Block(d, heads, d_ff, rng, cross=True) for _ in range(layers)]

    def _inputs(self, tokens):
        tokens = np.asarray(tokens)
        return self.embed[tokens] * np.sqrt(self.d) + sinusoidal_positions(len(tokens), self.d)

    def encode(self, tokens):
        """Encoder-only (BERT): no mask."""
        x = self._inputs(tokens)
        for b in self.encoder:
            x = b(x)
        return layer_norm(x)

    def decode(self, tokens):
        """Decoder-only (GPT): causal mask. Returns logits with the tied embedding."""
        x = self._inputs(tokens)
        mask = causal_mask(len(tokens))
        for b in self.decoder:
            x = b(x, mask=mask)
        return layer_norm(x) @ self.embed.T

    def seq2seq(self, source, target):
        """Encoder-decoder (2017 Transformer, T5): encode the source, then decode the target with cross-attention."""
        memory = self.encode(source)
        x = self._inputs(target)
        mask = causal_mask(len(target))
        for b in self.enc_dec:
            x = b(x, mask=mask, memory=memory)
        return layer_norm(x) @ self.embed.T
