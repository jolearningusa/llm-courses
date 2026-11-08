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

<a id="fair-optimizer-comparison"></a>
## Fair optimizer comparison — audit FND-2

Companion to optimizer slide 10 and schedule slide 7. Existing `best_lr`/`best_final` already search a grid; this extension makes the experiment contract and search cost explicit. The optimizer implementation and existing reference results are unchanged.

### Freeze the question before the search

Choose whether the question is best validation loss under a fixed **update budget** or under a fixed **time budget**. Equal updates are not equal runtime. Declare the optimizer as the variable of interest, learning rate as a tuning variable, and data/split/initialization/minibatch seed policies as controls. Momentum, beta, epsilon, schedule and decay settings are additional conditional parameters: freeze and disclose them, or allocate a declared budget to tune them. Do not call one shared numeric learning rate a fair test of every optimizer.

For a learned predictor, freeze train/validation/test IDs first. Fit each candidate only on training data. Select one setting **per optimizer** using the predeclared validation aggregation across fixed seeds; do not choose the luckiest individual seed. Lock the choice, any final-refit procedure and evaluation seed list, then evaluate on held-out data. Report every planned test run, including failures. Never use test losses to extend the grid, choose seeds or select checkpoints. If the protocol changes after looking at test results, the old test set has become development evidence and needs to be treated accordingly.

### Count the search as well as the selected fit

Constructed allocation from the slide: six learning rates × three seeds × at most 200 updates = **3,600 allotted tuning updates per optimizer**. Three final refits at at most 200 updates add **600 allotted updates**. These are caps, not measurements. Early stops and failures may consume fewer updates; validation, final evaluation, compilation and setup have additional costs. Record actual executed updates and time alongside the cap. The example does not require this exact grid or number of seeds.

| Field | Declare before running | Record after running |
|---|---|---|
| Scientific variable | Optimizer and supported configurations | Exact implementation/revision |
| Tuning variables | LR ranges, trial counts, any other tuned parameters | Every tried setting, seed and outcome |
| Fixed controls | Split, data, initialization/batch seeds, schedule rule | Deviations, preprocessing and environment |
| Allocation | Updates or time, failure/early-stop policy | Actual updates/time; failed/nonfinite trials |
| Selection | Validation metric and across-seed aggregation; tie rule | Chosen configuration and validation evidence |
| Held-out evaluation | Locked refit and evaluation protocol | All planned outcomes, uncertainty and failures |

### Reproduce a small optimization-only ledger

From this lab directory, run the snippet below. It uses the **existing** optimizer classes on the deterministic quadratic. Each optimizer gets the same three-candidate, 200-update cap and fresh state. This is an optimization toy with a known objective, **not** a train/validation/test generalization experiment. Repeating it under unused random-seed labels would not supply independent evidence. Finite loss can still be enormous; neither finite status nor the best grid point establishes a universal optimizer ranking.

```sh
python - <<'PY'
import numpy as np
from optimizers import SGD, Adam, quadratic, run
f, grad, start = quadratic(100)
for name, factory in [('SGD', SGD), ('Adam', Adam)]:
    for lr in (0.001, 0.01, 0.1):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                final, _ = run(factory(), f, grad, start, lr, 200)
                loss = f(final)
            status = 'finite' if np.isfinite(loss) else 'nonfinite'
            print(name, lr, status, loss)
        except FloatingPointError:
            print(name, lr, 'nonfinite: retain trial in ledger')
PY
```

The expected invariant is **six printed trial records**, not a hard-coded winner. Each method has 600 allotted updates; a failed run may terminate early, and this snippet does not instrument actual update counts or elapsed time. Save the output and environment. The selected objective must be evaluated at the returned final weights (`run` history contains pre-update values). For the graded optimizer behavior, keep running `python -m unittest` as before.

### Deliver and interpret

Submit the frozen protocol, all-trial ledger, per-optimizer validation-selected configuration, seed-level outcomes, held-out report, actual cost and one adverse case. The learned-data part is a student experiment protocol; no held-out model result is supplied here. The small snippet only executes the deterministic optimization ledger.

Prediction: A used 6 candidate settings and B used 60, but their selected final fits each used 200 updates. Is the entire search budget matched? **No.** Equal trial count alone is also insufficient if one setting trains much longer. Explain both the budget boundary and the limits of the explored ranges.

Retain the previous optimizer checkpoints: momentum's constant-gradient limit; Adam's first step with and without bias correction; its coordinate scaling; and AdamW versus L2-in-Adam. For learning-rate practice, continue to sweep on a log grid, retune after changing batch/model/normalization, and inspect instability rather than assuming the optimum always lies at one universal distance from it.

Primary reading: [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), experimental design and scientific/nuisance/fixed hyperparameters. The study diagram, allocation and ledger protocol are course-owned teaching material informed by that methodology.
