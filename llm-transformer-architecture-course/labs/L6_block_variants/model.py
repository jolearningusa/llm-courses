"""Lab L6: one small decoder-only language model whose block can be switched between the variants of
Part IV and Part III.

    norm:      "layer" (LayerNorm) or "rms" (RMSNorm)                 deck 17
    placement: "post" (2017 Transformer) or "pre" (GPT-2 onward)      deck 17
    ffn:       "relu" (4d hidden) or "swiglu" (parameter-matched)     deck 18
    n_kv_heads: n_heads for MHA, fewer for GQA, 1 for MQA            decks 12-13

Positions are learned absolute embeddings, so only the block changes between variants.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    ctx: int = 64
    d: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    norm: str = "layer"
    placement: str = "pre"
    ffn: str = "relu"


class RMSNorm(nn.Module):
    """a / RMS(a) * g, with RMS(a) = sqrt(mean(a^2) + eps). No mean subtraction, no bias (Zhang and Sennrich 2019)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


def make_norm(cfg: Config) -> nn.Module:
    return RMSNorm(cfg.d) if cfg.norm == "rms" else nn.LayerNorm(cfg.d)


def swiglu_hidden(d: int) -> int:
    """Hidden width that gives a SwiGLU FFN (3 matrices) the parameters of a ReLU FFN with 4d (2 matrices)."""
    return round(2 * 4 * d / 3)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.g, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.d // cfg.n_heads
        self.q = nn.Linear(cfg.d, self.h * self.dh, bias=False)
        self.kv = nn.Linear(cfg.d, 2 * self.g * self.dh, bias=False)
        self.o = nn.Linear(self.h * self.dh, cfg.d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.g, self.dh).permute(2, 0, 3, 1, 4)
        k = k.repeat_interleave(self.h // self.g, dim=1)          # each KV head serves a group of query heads
        v = v.repeat_interleave(self.h // self.g, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.kind = cfg.ffn
        if cfg.ffn == "swiglu":
            hidden = swiglu_hidden(cfg.d)
            self.w = nn.Linear(cfg.d, hidden, bias=False)          # gate
            self.v = nn.Linear(cfg.d, hidden, bias=False)          # value
        else:
            hidden = 4 * cfg.d
            self.w = nn.Linear(cfg.d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.w2(F.silu(self.w(x)) * self.v(x))         # (Swish(xW) * xV) W2
        return self.w2(F.relu(self.w(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.pre = cfg.placement == "pre"
        self.n1, self.n2 = make_norm(cfg), make_norm(cfg)
        self.attn, self.ffn = Attention(cfg), FFN(cfg)

    def forward(self, x):
        if self.pre:                                              # x + F(LN(x))
            x = x + self.attn(self.n1(x))
            return x + self.ffn(self.n2(x))
        x = self.n1(x + self.attn(x))                             # LN(x + F(x))
        return self.n2(x + self.ffn(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final = make_norm(cfg) if cfg.placement == "pre" else nn.Identity()   # Pre-LN needs a final norm
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.final(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
