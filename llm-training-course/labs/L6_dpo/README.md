# Lab L6: DPO, IPO and SimPO from log-probabilities

Used with decks 22 to 24. CPU only, Python 3.10+ and NumPy. Every run takes about a second.

Direct preference methods train on pairs with a loss that needs only log-probabilities: the policy's,
and for DPO and IPO the frozen reference model's. This lab writes the three losses with their gradients,
then trains a policy over six answers to one prompt on exact expected preferences. That makes it
possible to check DPO's central claim: its optimum is the KL-regularized RLHF policy, with no reward
model and no RL.

| File | What it holds | Deck |
|---|---|---|
| `losses.py` | DPO, IPO and SimPO losses and their derivatives with respect to the policy log-probabilities; DPO's implicit reward | 22, 23 |
| `bandit.py` | A softmax policy over K answers trained on the expected loss over all pairs, Bradley-Terry preferences, the closed-form RLHF optimum, and the deterministic-preference case | 22, 23 |

## Run

```bash
python -m unittest test_dpo.py   # about a second
python bandit.py                 # beta = 0.5
python bandit.py --beta 0.1
```

## Checks

- Each loss matches its formula, and its gradients match finite differences.
- At the reference policy the DPO loss is log 2 and the gradient weight is β/2.
- Repeating both answers k times leaves the SimPO loss unchanged (log-probabilities and lengths both scale
  by k); a margin of summed log-probabilities changes.
- Trained on Bradley-Terry preferences from a reward r, DPO reaches
  π\*(y) ∝ π<sub>ref</sub>(y) exp(r(y)/β), the RLHF optimum, to 10<sup>−6</sup> for β = 0.2, 0.5 and 2, and its
  implicit reward β log(π/π<sub>ref</sub>) equals r up to one constant (Rafailov et al.,
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) Eqs. 4–5).
- **Deterministic preferences:** when one answer always wins, DPO keeps driving the loser's probability
  toward zero whatever β is, while IPO stops at π(loser) = σ(−1/(2τ)), the value Azar et al. derive
  ([arXiv:2310.12036](https://arxiv.org/abs/2310.12036) §5.3.1).

## Reference results

`python bandit.py` on our machine (NumPy 2.5):

```text
DPO against the closed-form RLHF optimum, K = 6, beta = 0.5
answer   reward   pi_ref   pi_DPO      pi*
     0    1.304   0.1611   0.6550   0.6550
     1    0.947   0.1245   0.2479   0.2479
     2   -0.704   0.2695   0.0198   0.0198
     3   -1.265   0.1578   0.0038   0.0038
     4   -0.623   0.0831   0.0072   0.0072
     5    0.041   0.2040   0.0664   0.0664
max |log pi_DPO - log pi*| = 9.77e-15

Two answers, answer 0 always preferred, uniform reference
beta or tau  steps  DPO pi(1)  IPO pi(1) IPO target
        0.1   1000   3.47e-10     0.0067     0.0067
        0.1  10000   1.44e-20     0.0067     0.0067
        1.0   1000   1.00e-03     0.3775     0.3775
        1.0  10000   1.00e-04     0.3775     0.3775
```

## Tasks

1. Derive the DPO gradient by hand from the loss and check it against `dpo()`. Why does the weight
   σ(−z) matter? Replace it with a constant and train: what happens to the losing answers?
2. Sample a finite data set: three comparisons per pair from the Bradley-Terry probabilities. Train DPO
   and IPO on the empirical frequencies. Which drifts further from π\* in KL, and which answers drive it?
3. Give one answer zero reference probability support (a very negative reference log-prob) and no data.
   What does DPO do to it, and what would PPO with a KL penalty do (deck 24, Xu et al.)?
4. Write KTO's loss (deck 23) for unpaired examples marked desirable or undesirable, with the reference
   point z<sub>0</sub> held fixed, and train the bandit with it. How close does it get to π\*?
5. Write ORPO's odds-ratio term on top of a negative log-likelihood term. Why does it need no reference
   model, and what plays the reference's role?
6. Simulate length bias: make each answer's log-probability a per-token value times a length, let the
   preferred answers be longer on average, and compare what DPO and SimPO learn to favour.
