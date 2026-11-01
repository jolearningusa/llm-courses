"""Train the block variants of model.py on the same data, budget and seeds, and compare loss.

    python train.py                     # all variants, 3 seeds, a few minutes on a laptop CPU
    python train.py --steps 100 --seeds 1 --variants pre-ln post-ln

The data is generated: lines of additions such as "417+86=503;". The digits after "=" are
predictable from the line, the operands are not, so the loss has a floor above zero and a model
that learns to add reaches a lower loss than one that only learns the format.
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import time

import torch

from model import Config, TinyLM, count

VOCAB = "0123456789+=;"
STOI = {c: i for i, c in enumerate(VOCAB)}

VARIANTS = {
    "post-ln":        dict(placement="post", norm="layer", ffn="relu"),
    "pre-ln":         dict(placement="pre", norm="layer", ffn="relu"),
    "pre-rms":        dict(placement="pre", norm="rms", ffn="relu"),
    "pre-rms-swiglu": dict(placement="pre", norm="rms", ffn="swiglu"),
    "gqa-2":          dict(placement="pre", norm="rms", ffn="swiglu", n_kv_heads=2),
    "mqa":            dict(placement="pre", norm="rms", ffn="swiglu", n_kv_heads=1),
}


def make_text(n_lines: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = []
    for _ in range(n_lines):
        a, b = rng.randrange(1000), rng.randrange(1000)
        lines.append(f"{a}+{b}={a + b};")
    return "".join(lines)


def batches(data: torch.Tensor, ctx: int, batch: int, gen: torch.Generator):
    while True:
        ix = torch.randint(len(data) - ctx - 1, (batch,), generator=gen)
        x = torch.stack([data[i:i + ctx] for i in ix])
        y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
        yield x, y


def train(variant: str, seed: int, steps: int, lr: float, layers: int, eval_every: int = 25):
    torch.manual_seed(seed)
    cfg = Config(vocab=len(VOCAB), n_layers=layers, **VARIANTS[variant])
    model = TinyLM(cfg)
    train_data = torch.tensor([STOI[c] for c in make_text(20000, seed=1)])
    val_data = torch.tensor([STOI[c] for c in make_text(2000, seed=2)])
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)   # constant LR, no warm-up (deck 17)
    it = batches(train_data, cfg.ctx, 32, gen)
    val_x, val_y = next(batches(val_data, cfg.ctx, 128, torch.Generator().manual_seed(0)))
    curve = []
    for step in range(1, steps + 1):
        x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                _, vl = model(val_x, val_y)
            v = vl.item() if torch.isfinite(vl) else float("nan")
            curve.append((step, v))
    return count(model), curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--csv", default="curves.csv")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = []
    print(f"{'variant':16} {'params':>7}  final val loss (mean ± sd over {args.seeds} seeds)   time")
    for v in args.variants:
        t0, finals = time.time(), []
        for s in range(args.seeds):
            n, curve = train(v, s, args.steps, args.lr, args.layers)
            finals.append(curve[-1][1])
            rows += [(v, s, step, loss) for step, loss in curve]
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(f"{v:16} {n:7d}  {statistics.mean(finals):.3f} ± {sd:.3f}  {['%.3f' % f for f in finals]}  {time.time() - t0:5.1f}s")
    with open(args.csv, "w", newline="") as f:
        csv.writer(f).writerows([("variant", "seed", "step", "val_loss"), *rows])
    print(f"curves written to {args.csv}")


if __name__ == "__main__":
    main()
