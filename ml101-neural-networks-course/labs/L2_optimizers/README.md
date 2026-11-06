# Lab L2: optimizers and learning-rate schedules

Used with decks 08 and 09. CPU only, NumPy only.

`optimizers.py` implements each optimizer as a class with `step(params, grads, lr)`:

| Part | Names | Deck |
|---|---|---|
| Optimizers | `SGD`, `Momentum` (heavy ball, or `nesterov=True`), `AdaGrad`, `RMSProp`, `Adam` (with `bias_correction` and `weight_decay`, `decoupled=True` for AdamW) | 08 |
| Schedules | `constant`, `step_decay`, `cosine`, `warmup_cosine`, `inverse_sqrt` | 09 |
| Test problems | `quadratic` (curvatures 1 and κ on rotated axes), `rosenbrock`, `noisy_regression` | 08, 09 |
| Experiments | `best_lr` and `best_final` (search a log grid), `sgd_regression` (SGD under a schedule), `decay_only` (AdamW against L2 inside Adam) | 08, 09 |

## Run

```bash
python optimizers.py         # the numbers quoted in decks 08 and 09 (a few seconds)
python -m unittest           # 12 checks
```

## Tasks

1. On `quadratic(100)`, sweep the momentum β from 0 to 0.99 at each one's best learning rate. Plot steps to
   10<sup>−8</sup> against β and mark ((√κ − 1)/(√κ + 1))². Repeat for κ = 1000.
2. Set `angle=0` in `quadratic` so the valley is aligned with the axes, and race Adam against momentum again.
   Why does the alignment change the result so much for Adam and not for momentum?
3. Run RMSProp on the quadratic with a constant learning rate and with `cosine`. Why does it never reach
   10<sup>−8</sup> with the constant rate?
4. In `sgd_regression`, double the batch size several times and apply the linear scaling rule to the learning
   rate. Where does it stop working?
5. Implement gradient clipping by global norm and show what it does to a run that diverges without it.
6. Change `decay_only` to compare AdamW and L2-in-Adam when the loss gradient is not zero (for example on
   `noisy_regression`), and compare the final weight norms.

## Checks

Adam's bias-corrected first step equals the learning rate, and is 3.16 times larger without the correction;
AdamW shrinks all weights by the same factor while L2-in-Adam does not; AdaGrad's steps fall like 1/√t; gradient
descent diverges just above 2/λ<sub>max</sub>; momentum beats gradient descent at least threefold on κ = 100 and
Nesterov is no slower; Adam solves the Rosenbrock valley; every decaying schedule beats a constant learning rate
on noisy least squares; and each schedule has the expected shape.
