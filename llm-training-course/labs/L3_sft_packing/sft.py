"""Supervised fine-tuning of a tiny model on a synthetic instruction dataset.

    python sft.py                         # the four runs below, 3 seeds each
    python sft.py --runs packed --steps 1000

Runs (same data, steps, learning rate and seeds):
    padded         one example per row, loss on assistant tokens only
    packed         examples packed into 128-token rows with a block-diagonal mask, same loss
    all-tokens     packed, but the loss also covers the user's tokens
    packed-naive   packed with a plain causal mask and positions running across the row, as if
                   the row were one document: examples in a row can see each other

Reports exact-match accuracy of greedy answers on held-out conversations, and the share of
tokens in a batch that are padding.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from chat import ASSISTANT, END, USER, VOCAB, dataset, encode, next_token_example
from model import Config, TinyLM, masked_loss_sum
from packing import attention_mask, pack_batch, pad_batch

RUNS = {
    "padded": dict(pack=False, all_tokens=False, block=True),
    "packed": dict(pack=True, all_tokens=False, block=True),
    "all-tokens": dict(pack=True, all_tokens=True, block=True),
    "packed-naive": dict(pack=True, all_tokens=False, block=False),
}


def make_batch(convs, pack: bool, all_tokens: bool, block: bool, length: int = 128):
    exs = []
    for c in convs:
        x, y, m = next_token_example(c)
        if all_tokens:
            m = [1] * len(m)
        exs.append((x, y, m))
    ids, tgt, msk, doc, pos = pack_batch(exs, length) if pack else pad_batch(exs)
    if not block:
        pos = torch.arange(ids.size(1)).expand_as(ids)
    return ids, tgt, msk, pos, attention_mask(doc, block_diagonal=block), (doc == -1).float().mean().item()


@torch.no_grad()
def answer(model, user_text: str, max_new: int = 10) -> str:
    ids = [USER] + encode(user_text) + [END, ASSISTANT]
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor([ids]))
        nxt = int(logits[0, -1].argmax())
        if nxt == END:
            break
        out.append(nxt)
        ids.append(nxt)
    return "".join(VOCAB[i] for i in out)


def exact_match(model, convs) -> float:
    hits = sum(answer(model, c[0][1]) == c[1][1] for c in convs)
    return hits / len(convs)


def train(run: str, seed: int, steps: int, batch: int = 32, lr: float = 3e-3) -> tuple[float, float]:
    torch.manual_seed(seed)
    cfg = RUNS[run]
    model = TinyLM(Config(vocab=len(VOCAB)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    data = dataset(steps * batch, seed=seed)
    pads = []
    for s in range(steps):
        ids, tgt, msk, pos, mask, pad_share = make_batch(data[s * batch:(s + 1) * batch], **cfg)
        pads.append(pad_share)
        total, count = masked_loss_sum(model(ids, pos, mask), tgt, msk)
        loss = total / count            # mean over the batch's trained tokens
        opt.zero_grad()
        loss.backward()
        opt.step()
    held_out = [c[:2] for c in dataset(200, seed=10_000 + seed, max_turns=1)]
    model.eval()
    return exact_match(model, held_out), statistics.mean(pads)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=list(RUNS))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    for run in a.runs:
        t0 = time.time()
        res = [train(run, seed, a.steps) for seed in range(a.seeds)]
        acc = [r[0] for r in res]
        print(f"{run:13s} exact match {statistics.mean(acc):.3f} ± {statistics.stdev(acc) if len(acc) > 1 else 0:.3f}"
              f"   padding {res[0][1]:5.1%}   {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()
