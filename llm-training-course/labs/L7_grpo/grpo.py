"""GRPO and Dr. GRPO on the toy arithmetic task (decks 26 and 28).

    python grpo.py                  # SFT warm-up, then GRPO and Dr. GRPO from the same start
    python grpo.py --steps 400

GRPO (Shao et al. 2024): sample G responses per prompt, advantage = (r - mean) / std within the group,
PPO's clipped ratio per token, each response's token losses averaged over its own length, then averaged
over responses. No value model.

Dr. GRPO (Liu et al. 2025) removes the two normalizations that bias it: the division by the group's
standard deviation, and the division by each response's length (replaced by one constant, the maximum
length). The KL term is left out, as in DAPO and Dr. GRPO.
"""
import argparse
import random

import torch

from task import prompts, split, reward, target, MAX_LEN, EOS
from policy import Policy


def group_advantages(r, std_normalize=True, eps=1e-6):
    """r: (P, G) rewards for P prompts with G responses each. Each row of the result sums to zero."""
    adv = r - r.mean(1, keepdim=True)
    if std_normalize:
        adv = adv / (r.std(1, keepdim=True) + eps)
    return adv


def token_weights(mask, length_normalize=True):
    """Weight of each token in the loss. GRPO: 1/|o_i| per response, then the mean over responses.
    Dr. GRPO: one constant, 1/MAX_LEN, for every token."""
    n = mask.shape[0]
    if length_normalize:
        return mask / mask.sum(1, keepdim=True) / n
    return mask / MAX_LEN / n


def policy_loss(lp_new, lp_old, adv, mask, eps=0.2, length_normalize=True):
    """Clipped surrogate (to be minimized) for N responses; adv is (N,), the rest (N, T)."""
    ratio = torch.exp(lp_new - lp_old)
    A = adv[:, None]
    per_token = -torch.min(ratio * A, torch.clamp(ratio, 1 - eps, 1 + eps) * A)
    return (per_token * token_weights(mask, length_normalize)).sum()


def sft(policy, data, steps=300, lr=3e-3, seed=0):
    """Warm-up on a few prompts, with 0 to 3 filler tokens before the answer."""
    rng = random.Random(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        batch = [rng.choice(data) for _ in range(32)]
        toks = [target(p, rng.randint(0, 3)) for p in batch]
        tokens = torch.tensor([t + [EOS] * (MAX_LEN - len(t)) for t in toks])
        a = torch.tensor([p[0] for p in batch])
        b = torch.tensor([p[1] for p in batch])
        lp, mask = policy.logprobs(a, b, tokens)
        loss = -(lp * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(policy, ps, samples=8, seed=0):
    """Mean reward over sampled responses, and the mean length of correct and incorrect ones."""
    g = torch.Generator().manual_seed(seed)
    a = torch.tensor([p[0] for p in ps]).repeat_interleave(samples)
    b = torch.tensor([p[1] for p in ps]).repeat_interleave(samples)
    tokens = policy.sample(a, b, generator=g)
    _, mask = policy.logprobs(a, b, tokens)
    lens = mask.sum(1)
    r = torch.tensor([reward((int(x), int(y)), t.tolist()) for x, y, t in zip(a, b, tokens)])
    ok, bad = lens[r == 1], lens[r == 0]
    return r.mean().item(), (ok.mean().item() if len(ok) else 0.0), (bad.mean().item() if len(bad) else 0.0)


def train(policy, ps, steps=300, prompts_per_step=16, G=8, lr=3e-3, inner=2, dr=False, seed=0):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        batch = [rng.choice(ps) for _ in range(prompts_per_step)]
        a = torch.tensor([p[0] for p in batch]).repeat_interleave(G)
        b = torch.tensor([p[1] for p in batch]).repeat_interleave(G)
        tokens = policy.sample(a, b, generator=g)
        r = torch.tensor([reward((int(x), int(y)), t.tolist()) for x, y, t in zip(a, b, tokens)])
        adv = group_advantages(r.view(-1, G), std_normalize=not dr).view(-1)
        with torch.no_grad():
            lp_old, mask = policy.logprobs(a, b, tokens)
        for _ in range(inner):
            lp_new, _ = policy.logprobs(a, b, tokens)
            loss = policy_loss(lp_new, lp_old, adv, mask, length_normalize=not dr)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    sft_prompts, _ = split(args.seed)
    base = Policy()
    sft(base, sft_prompts, seed=args.seed)
    ps = prompts()
    acc, lo, lb = evaluate(base, ps)
    print(f"after SFT on {len(sft_prompts)} prompts: mean reward {acc:.3f} "
          f"(length: correct {lo:.2f}, incorrect {lb:.2f})")
    state = {k: v.clone() for k, v in base.state_dict().items()}
    for name, dr in (("GRPO", False), ("Dr. GRPO", True)):
        torch.manual_seed(args.seed)
        pol = Policy()
        pol.load_state_dict(state)
        train(pol, ps, steps=args.steps, dr=dr, seed=args.seed)
        acc, lo, lb = evaluate(pol, ps)
        print(f"{name:>9} after {args.steps} steps: mean reward {acc:.3f} "
              f"(length: correct {lo:.2f}, incorrect {lb:.2f})")


if __name__ == "__main__":
    main()
