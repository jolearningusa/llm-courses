# Lab L0: linear and logistic models from scratch

Used with decks 01 to 04. CPU only, NumPy only.

`linear_models.py` implements, with every gradient written by hand:

| Part | Functions | Deck |
|---|---|---|
| Least squares | `fit_least_squares` (normal equations, optional ridge), `mse`, `mse_grad` | 01 |
| Gradient descent | `gradient_descent` (full batch), `sgd_linear` (mini-batch, shuffled epochs) | 01, 03 |
| Classification | `sigmoid`, `binary_cross_entropy`, `softmax`, `log_softmax`, `softmax_cross_entropy` and their gradients | 02 |
| Checking | `numerical_grad` (central differences), `relative_error` | 03 |
| Generalization | `polynomial_features`, `degree_sweep` (training and validation error by degree) | 04 |

## Run

```bash
python linear_models.py      # the numbers quoted in decks 01, 02 and 04
python -m unittest           # 15 checks, under a second
```

## Tasks

1. Derive `mse_grad` and `softmax_cross_entropy_grad` on paper, then confirm them with `numerical_grad`.
2. Find the largest learning rate for which `gradient_descent` on the regression data still converges.
   Compare it with 2 / λ<sub>max</sub> of (2/n) XᵀX (the test `test_too_large_learning_rate_diverges` uses 2.2 / λ<sub>max</sub>).
3. Run `sgd_linear` with batch sizes 1, 10 and 200 at the same learning rate. Plot the loss per epoch.
   Which is fastest per epoch, and which per gradient evaluation?
4. Remove the `Z - Z.max(...)` line from `softmax` and call it on logits of size 1000. What happens?
5. Extend `degree_sweep` to degrees 0 to 14 and plot training and validation MSE. Then sweep the ridge
   λ at degree 12 from 10<sup>−8</sup> to 1 and find the best value on the validation set.
6. Train logistic regression on two well-separated blobs for 10, 100 and 10,000 steps and print ‖w‖.
   Explain the growth (deck 02, slide 8), then add an L2 penalty and repeat.

## Checks

The closed form recovers the generating weights; gradient descent and SGD reach the closed form; a learning
rate above 2 / λ<sub>max</sub> diverges; every analytic gradient matches finite differences; the sigmoid and
softmax are stable for huge inputs; two-class softmax equals the sigmoid; zero weights cost exactly log k;
softmax regression separates three blobs; training error falls with degree while validation error is U-shaped;
and ridge fixes the degree-12 fit.
