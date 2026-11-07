# Lab L5: a Bradley-Terry reward model and PPO with a KL penalty

Used with decks 18 to 20. CPU only, Python 3.10+ and PyTorch 2. Every run takes a few seconds.

RLHF fits a reward model to pairwise preferences, then optimizes a policy against it with PPO while a KL
penalty keeps it near the reference model. This lab does all three in a toy world small enough to know
the right answers: 400 candidate responses, each a point in two dimensions, and a hidden **gold** reward
that stands for what people want. The gold reward likes one direction but punishes extremes. The
reference policy, standing in for the SFT model, mostly produces moderate responses, so that is where the
preference data comes from.

| File | What it holds | Deck |
|---|---|---|
| `preferences.py` | The toy world, preference pairs labeled by Bradley-Terry on the gold reward, the Bradley-Terry loss, a linear reward model and its held-out pair accuracy | 18 |
| `ppo.py` | PPO's clipped objective, generalized advantage estimation, the KL-penalized reward, and PPO on a policy that tilts the reference policy | 19, 20 |
| `rlhf.py` | Fits the reward model, then runs PPO against it at several KL weights β and reports the proxy and gold rewards | 18–20 |

## Run

```bash
python -m unittest test_rlhf.py   # about 3 seconds
python rlhf.py                    # 2,000 comparisons, β from 1 to 0.1
python rlhf.py --pairs 30         # a reward model from 30 comparisons
```

## Checks

- The Bradley-Terry loss is −log σ(r<sub>w</sub> − r<sub>l</sub>), and the two orders of a pair have
  probabilities that sum to one.
- The reward model trained on 2,000 comparisons orders over 85% of held-out pairs as the gold reward
  does, and one trained on 30 comparisons does worse.
- The clipped objective equals min(ρA, clip(ρ, 1 − ε, 1 + ε)A), and its gradient is zero once the ratio
  is past the clip in the direction the advantage favours.
- Twenty epochs on one batch at a large learning rate: with clipping the policy moves less than 0.3 nats
  of KL; without it, over 30 times further.
- The KL-regularized optimum of a linear reward w·φ is the reference policy tilted by θ = w/β; PPO
  reaches it within 2% at β = 0.5.
- **Overoptimization:** as β shrinks, KL to the reference and the proxy reward rise together, while the
  gold reward peaks at an intermediate KL and then falls below its value at β = 1 (deck 20; Gao et al.,
  [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)).
- GAE with λ = 0 is the one-step TD error; with λ = 1 (and γ = 1) it is the return minus the value.

## Reference results

`python rlhf.py` on our machine (PyTorch 2.15, CPU):

```text
reward model from 2000 comparisons: w = [0.915 0.41 ], held-out pair accuracy 0.908

reference policy: proxy -0.078, gold -0.561
  beta            theta           w/beta  KL(pi||ref)    proxy     gold
  1.00      [0.91 0.41]      [0.91 0.41]        0.260    0.438   -0.087
  0.50      [1.83 0.82]      [1.83 0.82]        1.002    0.936    0.149
  0.30      [3.05 1.37]      [3.05 1.37]        2.517    1.508    0.113
  0.20      [4.57 2.05]      [4.57 2.05]        5.016    2.112   -0.263
  0.10      [8.96 4.1 ]      [9.15 4.1 ]       10.148    2.869   -1.240
```

The reward model is linear and the gold reward is not: it is right near the reference policy, where the
data is, and keeps rising in the liked direction where there is none. The policy follows it there.

## Tasks

1. Plot the gold reward against √KL for many values of β. Fit Gao et al.'s best-of-n form d(α − βd) and
   their RL form d(α − β log d). Which fits this toy better?
2. Run with `--pairs 30`. How does a worse reward model change where the gold reward peaks?
3. Give the reward model the feature |φ|² as well. Does overoptimization go away? What does that say about
   reward models that can represent the true preference?
4. Turn off the advantage baseline (use R instead of R − mean R). How much noisier is training?
5. Set ε = 0.05 and ε = 1.0 in `ppo_step`. How many iterations does PPO need to reach θ = w/β, and how
   large is each step's KL?
6. Write best-of-n sampling: draw n responses from the reference policy and keep the one the reward model
   likes best. Compare its gold reward with PPO's at the same KL, using log n − (n − 1)/n as the KL of
   best-of-n (deck 20).
7. Replace the gold labels with a biased "AI labeler" that prefers the first answer of a pair 20% of the
   time regardless of quality. How much does held-out pair accuracy drop? Then score each pair in both
   orders and average, as RLAIF does (deck 21). How much comes back?
