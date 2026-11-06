"""A small decoder that takes explicit positions and an attention mask, so it can run packed rows."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int
    ctx: int = 128
    d: int = 64
    n_layers: int = 2
    n_heads: int = 4


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.n1 = nn.RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.n2 = nn.RMSNorm(cfg.d)
        self.up = nn.Linear(cfg.d, 4 * cfg.d, bias=False)
        self.down = nn.Linear(4 * cfg.d, cfg.d, bias=False)

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.h, d // self.h).transpose(1, 3).unbind(2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)

    def forward(self, idx, pos=None, mask=None):
        """idx, pos: [b, t]; mask: boolean [b, 1, t, t] or None for plain causal attention."""
        t = idx.size(1)
        if pos is None:
            pos = torch.arange(t).expand_as(idx)
        if mask is None:
            mask = torch.ones(t, t, dtype=torch.bool).tril()
        x = self.tok(idx) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm(x))


def masked_loss_sum(logits, targets, mask):
    """Sum of next-token cross-entropy over tokens where mask is 1, and the number of such tokens."""
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    m = mask.reshape(-1)
    return (ce * m).sum(), m.sum()
