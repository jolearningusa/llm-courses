# Lab L8: Distillation with a temperature, forward and reverse KL, on-policy

Used with decks 29 and 30. CPU only, Python 3.10+ and NumPy. Every run takes about a second.

Distillation trains a student on a teacher's distribution instead of on one-hot labels. The lab looks at
the two choices that separate the methods in the decks. The first is the temperature of Hinton-style
knowledge distillation and why the soft loss is multiplied by T². The second is the direction of the KL
divergence: supervised KD on teacher outputs minimizes the forward KL, while MiniLLM and on-policy GKD
minimize the reverse KL on the student's own samples. A one-Gaussian student fitted to a two-mode teacher
shows the difference.

| File | What it holds | Deck |
|---|---|---|
| `kd.py` | Softened softmax, the soft cross-entropy and its gradient (q − p)/T, and the combined loss α·T²·soft + (1 − α)·hard | 29 |
| `modes.py` | A two-Gaussian teacher on a grid, a Gaussian student, forward KL, reverse KL and the generalized JSD, their exact gradients, the on-policy (policy-gradient) estimate, and an Adam loop | 30 |

## Run

```bash
python -m unittest test_kd.py   # about a second
python kd.py
python modes.py
```

## Checks

- The soft cross-entropy gradient matches finite differences at T = 1, 3 and 10.
- At high temperature, T² times the gradient approaches (z − v)/N for zero-mean logits: distillation
  becomes logit matching (Hinton et al., [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) §2.1).
- The generalized JSD of GKD, divided by β(1 − β), tends to the forward KL as β → 0 and to the reverse KL
  as β → 1 (Agarwal et al., [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)).
- Both exact KL gradients match finite differences, and the on-policy estimate from student samples
  averages to the exact reverse-KL gradient.
- **Forward KL covers both modes:** from three starts the student ends near the mixture mean (−0.6) with
  σ wider than the mixture's own spread.
- **Reverse KL picks one mode:** started right of centre the student settles on the +3 mode with σ = 0.8,
  started left of centre on the −3 mode. On-policy training from samples reaches the same +3 mode.

## Reference results

`python modes.py` on our machine (NumPy 2.5):

```text
teacher: modes at -3 and +3 (weights 0.6 and 0.4), std 0.8 each; mixture mean -0.6, std 3.05
student: one Gaussian, starting at mu=1.5, sigma=0.61 unless stated
                     objective      mu   sigma  KL(p||q)  KL(q||p)
    forward KL (supervised KD)  -0.660   3.202     0.653     1.604
             reverse KL, exact   2.999   0.802    16.115     0.916
 reverse KL, on-policy samples   2.998   0.787    16.746     0.916
  reverse KL, start at mu=-1.5  -2.999   0.801    10.538     0.511
        reverse KL, start wide  -0.492   2.761     0.671     1.525
```

Two things in this table are worth reading slowly. Reverse KL chose the lighter mode (weight 0.4) because
it was nearer the start: it is not a search for the best mode, and the −3 solution has the lower reverse
KL. And started wide (σ = 1.65), reverse KL stalls in a broad local optimum between the modes, where the
gradient is zero. Mode seeking is a tendency of the objective near a mode, not a guarantee from any start.
The forward-KL σ (3.2) is a little above the mixture's 3.05 because the student is normalized on a grid
cut at ±8.

## Tasks

1. Train a small classifier (for example on scikit-learn's digits) as a teacher, then a smaller student
   with `kd_loss` at T = 1, 4 and 20 and α from 0 to 1. Where does the T² factor matter, and what happens
   if you leave it out at T = 20?
2. Remove one class from the student's training set, as Hinton et al. did with the digit 3. How much of
   that class does the distilled student still get right from soft targets alone?
3. Fit the student with the JSD(β) objective for β = 0.1, 0.5 and 0.9 (write its gradient, or use finite
   differences). Where does the fitted Gaussian move as β grows?
4. Implement MiniLLM's teacher-mixed sampling: draw each on-policy sample from α·p + (1 − α)·q and
   reweight with the importance ratio q/(α·p + (1 − α)·q). Does it escape the wide local optimum?
5. Change the teacher weights to 0.9 and 0.1 and the student start to the light mode. What does each
   objective's answer cost in the other's divergence, and which student would you rather sample from?
