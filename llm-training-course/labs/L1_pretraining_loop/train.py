"""Pre-train the same tiny model with different schedules and optimizers, same data, budget and seeds.

    python train.py                                 # four runs x three seeds, a few minutes on a laptop CPU
    python train.py --steps 200 --seeds 1 --runs adamw-wsd muon-wsd
    python train.py --lr 3e-2 --runs adamw-constant adamw-constant-qk-z   # stability at a high LR (deck 03)

The data is generated lines of additions such as "417+86=503;". The operands are random, so the
loss has a floor above zero; a model that learns to add gets below a model that only learns the format.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics

import torch

from model import Config, TinyLM, lm_loss
from optim import AdamW, Muon
from schedules import SCHEDULES

VOCAB = "0123456789+=;"
STOI = {c: i for i, c in enumerate(VOCAB)}

# name: (optimizer for hidden matrices, schedule, qk_norm, z-loss coefficient)
RUNS = {
    "adamw-constant":      ("adamw", "constant", False, 0.0),
    "adamw-cosine":        ("adamw", "cosine", False, 0.0),
    "adamw-wsd":           ("adamw", "wsd", False, 0.0),
    "muon-wsd":            ("muon", "wsd", False, 0.0),
    "adamw-constant-qk-z": ("adamw", "constant", True, 1e-4),
}


def make_text(n_lines: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n_lines):
        a, b = rng.randrange(1000), rng.randrange(1000)
        out.append(f"{a}+{b}={a + b};")
    return "".join(out)


def batch(data: torch.Tensor, ctx: int, size: int, gen: torch.Generator):
    ix = torch.randint(len(data) - ctx - 1, (size,), generator=gen)
    return (torch.stack([data[i:i + ctx] for i in ix]),
            torch.stack([data[i + 1:i + ctx + 1] for i in ix]))


def build_optimizers(model: TinyLM, hidden_opt: str, lr: float, wd: float):
    """Hidden 2-D weights get AdamW or Muon with weight decay; embeddings, head and norms get AdamW
    without weight decay (OLMo 2 removes weight decay from embeddings, arXiv:2501.00656 section 3.4.2)."""
    hidden = model.hidden_matrices()
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opts = [AdamW(rest, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)]
    if hidden_opt == "muon":
        opts.append(Muon(hidden, lr=lr, weight_decay=wd))
    else:
        opts.append(AdamW(hidden, lr=lr, betas=(0.9, 0.95), weight_decay=wd))
    return opts


@torch.no_grad()
def evaluate(model, data, ctx, gen, n=8):
    losses = [lm_loss(model(x), y)[1].item() for x, y in (batch(data, ctx, 64, gen) for _ in range(n))]
    return sum(losses) / len(losses)


def train(run: str, seed: int, steps: int, lr: float, wd: float = 0.1, warmup_frac: float = 0.05,
          log_every: int = 50):
    hidden_opt, sched_name, qk_norm, z_coef = RUNS[run]
    torch.manual_seed(seed)
    cfg = Config(vocab=len(VOCAB), qk_norm=qk_norm)
    model = TinyLM(cfg)
    train_data = torch.tensor([STOI[c] for c in make_text(20000, seed=1)])
    val_data = torch.tensor([STOI[c] for c in make_text(2000, seed=2)])
    gen = torch.Generator().manual_seed(seed)
    opts = build_optimizers(model, hidden_opt, lr, wd)
    sched = SCHEDULES[sched_name]
    warmup = max(1, int(warmup_frac * steps))
    curve = []
    for t in range(steps):
        scale = sched(t, steps, warmup=warmup)
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = lr * scale
        x, y = batch(train_data, cfg.ctx, 32, gen)
        loss, ce = lm_loss(model(x), y, z_coef)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        if not math.isfinite(ce.item()):
            return float("nan"), curve
        if (t + 1) % log_every == 0 or t + 1 == steps:
            curve.append((t + 1, scale, evaluate(model, val_data, cfg.ctx, torch.Generator().manual_seed(0))))
    return curve[-1][2], curve


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", default=["adamw-constant", "adamw-cosine", "adamw-wsd", "muon-wsd"],
                   choices=list(RUNS))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--csv", default="curves.csv")
    a = p.parse_args()
    rows = []
    print(f"{'run':<22}{'val loss, mean':>16}{'stdev':>8}")
    for run in a.runs:
        finals = []
        for seed in range(a.seeds):
            final, curve = train(run, seed, a.steps, a.lr)
            finals.append(final)
            rows += [(run, seed, s, lr_scale, loss) for s, lr_scale, loss in curve]
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        print(f"{run:<22}{statistics.mean(finals):>16.4f}{sd:>8.4f}")
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "step", "lr_scale", "val_loss"])
        w.writerows(rows)
    print(f"curves written to {a.csv}")


if __name__ == "__main__":
    main()
