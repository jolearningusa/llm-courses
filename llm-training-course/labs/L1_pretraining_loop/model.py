"""A small decoder-only character model with the two stability switches of deck 03.

qk_norm: RMS-normalize queries and keys before the dot product (Wortsman et al., arXiv:2309.14322 3.1.1).
z_loss:  add coef * log(Z)^2 to the loss, Z the softmax normalizer (PaLM, arXiv:2204.02311 section 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    ctx: int = 32
    d: int = 64
    n_layers: int = 2
    n_heads: int = 4
    qk_norm: bool = False


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n1 = nn.RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.n2 = nn.RMSNorm(cfg.d)
        self.up = nn.Linear(cfg.d, 4 * cfg.d, bias=False)
        self.down = nn.Linear(4 * cfg.d, cfg.d, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        h = self.cfg.n_heads
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, h, d // h).transpose(1, 3).unbind(2)
        if self.cfg.qk_norm:
            q, k = F.rms_norm(q, (d // h,)), F.rms_norm(k, (d // h,))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def hidden_matrices(self):
        """The 2-D weights inside the blocks: the ones Muon updates (deck 04)."""
        return [p for blk in self.blocks for p in blk.parameters() if p.ndim == 2]


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, z_coef: float = 0.0):
    """Mean next-token cross-entropy, plus the optional z-loss. Returns (total, cross-entropy)."""
    flat = logits.reshape(-1, logits.size(-1))
    ce = F.cross_entropy(flat, targets.reshape(-1))
    if z_coef == 0.0:
        return ce, ce
    log_z = torch.logsumexp(flat, dim=-1)
    return ce + z_coef * (log_z ** 2).mean(), ce
