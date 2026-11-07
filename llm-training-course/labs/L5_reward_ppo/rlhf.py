"""Lab L5 driver: fit a Bradley-Terry reward model, then run PPO against it at several KL weights.

    python rlhf.py                    # reward model, then a sweep over beta
    python rlhf.py --pairs 30         # fewer comparisons: a worse reward model
"""
import argparse
import torch

from preferences import make_world, gold_reward, sample_preferences, train_rm, pair_accuracy
from ppo import train_policy, kl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=2000)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1])
    args = p.parse_args()
    torch.manual_seed(0)

    phi, ref_logp = make_world()
    w, l = sample_preferences(phi, ref_logp, args.pairs, seed=1)
    rm = train_rm(phi, w, l)
    ha, hb = sample_preferences(phi, ref_logp, 2000, seed=2, label_noise=False)
    print(f"reward model from {args.pairs} comparisons: w = {rm.w.detach().numpy().round(3)}, "
          f"held-out pair accuracy {pair_accuracy(rm, phi, ha, hb):.3f}")

    proxy = rm(phi).detach()
    gold = gold_reward(phi)
    ref_p = ref_logp.exp()
    print(f"\nreference policy: proxy {(ref_p * proxy).sum():.3f}, gold {(ref_p * gold).sum():.3f}")
    print(f"{'beta':>6} {'theta':>16} {'w/beta':>16} {'KL(pi||ref)':>12} {'proxy':>8} {'gold':>8}")
    for beta in args.betas:
        theta, logp = train_policy(phi, ref_logp, proxy, iters=args.iters, beta=beta)
        pr = logp.exp()
        print(f"{beta:6.2f} {str(theta.numpy().round(2)):>16} {str((rm.w.detach() / beta).numpy().round(2)):>16} {kl(logp, ref_logp):12.3f} {(pr * proxy).sum():8.3f} {(pr * gold).sum():8.3f}")


if __name__ == "__main__":
    main()
