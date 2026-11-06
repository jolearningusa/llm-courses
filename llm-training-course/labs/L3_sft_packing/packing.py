"""Batches for SFT: padded one example per row, or several examples packed into each row.

Packing (deck 15) removes padding by concatenating examples. It is only exact when each token can
attend to earlier tokens of its own example (a block-diagonal causal mask) and positions restart at
0 for each example. Then a packed batch gives the same per-token losses as an unpacked one.
"""
from __future__ import annotations

import torch

from chat import PAD


def pad_batch(examples):
    """One example per row, right-padded. Returns ids, targets, mask, doc ids, positions."""
    t = max(len(x) for x, _, _ in examples)
    rows = [_row([ex], t) for ex in examples]
    return tuple(torch.stack(c) for c in zip(*rows))


def pack_batch(examples, length: int):
    """Greedy packing in order: start a new row when the next example does not fit."""
    rows, cur, used = [], [], 0
    for ex in examples:
        n = len(ex[0])
        if n > length:
            raise ValueError("example longer than the packed row")
        if used + n > length:
            rows.append(cur)
            cur, used = [], 0
        cur.append(ex)
        used += n
    if cur:
        rows.append(cur)
    built = [_row(r, length) for r in rows]
    return tuple(torch.stack(c) for c in zip(*built))


def _row(examples, length: int):
    ids, tgt, msk, doc, pos = [], [], [], [], []
    for d, (x, y, m) in enumerate(examples):
        ids += x
        tgt += y
        msk += m
        doc += [d] * len(x)
        pos += list(range(len(x)))
    pad = length - len(ids)
    ids += [PAD] * pad
    tgt += [PAD] * pad
    msk += [0] * pad
    doc += [-1] * pad
    pos += [0] * pad
    return (torch.tensor(ids), torch.tensor(tgt), torch.tensor(msk, dtype=torch.float),
            torch.tensor(doc), torch.tensor(pos))


def attention_mask(doc: torch.Tensor, block_diagonal: bool = True) -> torch.Tensor:
    """Boolean [batch, 1, t, t]: True where query i may attend to key j.

    Causal, and (if block_diagonal) only inside the same example. Padding attends to itself so
    that no softmax row is empty; its loss is masked out anyway.
    """
    t = doc.size(1)
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    allowed = causal & (doc[:, :, None] != -1) & (doc[:, None, :] != -1)
    if block_diagonal:
        allowed &= doc[:, :, None] == doc[:, None, :]
    allowed |= torch.eye(t, dtype=torch.bool)
    return allowed[:, None]
