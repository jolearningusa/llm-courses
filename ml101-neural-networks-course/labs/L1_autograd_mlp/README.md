# Lab L1: an autograd engine and multilayer perceptrons

Used with decks 05 to 07. CPU only, NumPy only.

| File | What it contains | Deck |
|---|---|---|
| `autograd.py` | A `Tensor` that records how it was computed, and `backward()`, reverse-mode differentiation in reverse topological order. Operations: `+`, `*`, `@`, `relu`, `tanh`, `sigmoid`, `sum`, `mean` and a fused softmax `cross_entropy`. | 06 |
| `mlp.py` | `MLP` with ReLU or tanh layers and `small`, `xavier` or `he` initialization; `train` (full or mini-batch SGD); XOR and three-spiral data; `activation_stats`, which measures the scale of activations and gradients at every layer of a deep network at initialization. | 05, 07 |

## Run

```bash
python mlp.py                # the numbers quoted in decks 05 and 07 (about 10 s)
python -m unittest           # autograd/MLP and diagnostic checks, a few seconds
```

## Tasks

1. Read `Tensor.__matmul__` and `_unbroadcast`. Derive ∂L/∂A and ∂L/∂B for Y = AB from the
   elementwise definition and check them against the code.
2. Add a `Tensor.exp()` and a `Tensor.log()` and build cross-entropy from them instead of the fused node.
   Check its gradient. Then feed it logits of size 1000 and compare with the fused version.
3. Train `MLP([2, h, 3])` on the spirals for h = 2, 4, 8, 16 and 64. Plot accuracy against h. Hold out a
   third of the points and repeat: where does the validation accuracy stop improving?
4. Change `activation_stats` to depth 30 and compare `xavier` and `he` with ReLU. Then try tanh with `he`:
   what goes wrong, and why?
5. Set every weight of a hidden layer to the same constant and train on XOR. Show that the hidden units
   stay identical.
6. Replace ReLU by a "leaky" ReLU (slope 0.01 for negative inputs) and adjust the He variance for it.

## Checks

Gradients add when a value is reused; a broadcast bias gets the sum of its gradients; every parameter
gradient of a two-hidden-layer network matches finite differences, for ReLU and tanh; a linear model fails
XOR and one hidden layer solves it; an MLP solves the spirals that softmax regression cannot; mini-batch SGD
trains too; small initialization makes activations vanish below 10<sup>−6</sup> by layer 10; He initialization
keeps activations and gradients within a factor of 2 across ten ReLU layers; and Xavier suits tanh better than ReLU.

<a id="gradient-diagnostics"></a>
## Gradient diagnostics: a check can fail for different reasons

Audit FND-1 extension, paired with deck 06 slide 10 and deck 19 slides 4/10. The existing autograd/MLP code is unchanged. `gradient_diagnostics.py` evaluates the existing `Tensor` on a cubic, checks deliberately wrong gradients, sweeps the centered-difference step, and separates ReLU kinks from smooth numerical checks. `test_gradient_diagnostics.py` supplies six acceptance/negative-control tests.

Before running, predict: does decreasing h always improve the approximation? Does crossing a ReLU kink imply a backward bug? Can a relative error of 1 coexist with a tiny absolute error?

```sh
python gradient_diagnostics.py --output my-diagnostics.json
python -m unittest test_gradient_diagnostics -v
python -m unittest -v
```

Use a new output path: the command refuses to overwrite an existing JSON file. NumPy is the only normal runtime dependency, as in the existing lab. For optional plots, install Matplotlib in an isolated environment, then run:

```sh
python gradient_diagnostics.py --plot-dir my-figures
```

The checked-in [report](figures-report.json) and [smooth-step](figures/smooth_step.svg) / [near-zero](figures/near_zero.svg) SVGs were generated on Python 3.9.6, NumPy 2.0.2 and Matplotlib 3.9.4. Last digits and the best grid point can depend on platform/version. These are executed toy arithmetic results, not trained-model or GPU benchmarks.

### Expected outcomes and interpretation

- For `f(x)=x³` at x=1, the existing autograd yields 3. With h=1e−5, the declared smooth-fixture tolerance accepts it and rejects `wrong_sign` (−3) and `missing_contribution` (2). At h=1e−20, the float64 perturbations are lost and the finite-difference result becomes 0. Smaller is therefore not always better. For this polynomial, exact-real centered difference is `3x²+h²`; roundoff adds a separate effect.
- ReLU at x=−1e−4 has derivative 0. h=1e−3 crosses zero and gives 0.45; h=1e−5 stays on the flat branch and gives 0. At x=0 the classical derivative is undefined. A library's selected subgradient and a symmetric difference of 0.5 answer different questions.
- Near zero, adding 1e−12 to a zero reference yields absolute error 1e−12 and raw relative error 1. Our plot uses the max-magnitude denominator, with both-zero error defined as zero. Existing MLP tests use a different sum-based denominator and floor; do not silently treat their thresholds as interchangeable.
- `matches()` uses `abs(candidate-reference) <= atol + rtol*abs(reference)` with 1e−8/1e−6 **for the smooth cubic controls**. The near-zero example demonstrates the importance of an absolute scale; it is not a prescribed global policy. Inspect raw values, dtype, smoothness and scale before choosing an operation-specific contract. Nonfinite values are not passes.

Submit predictions, raw JSON, both error plots, an accepted smooth control, a rejected mutant and one minimized kink/step-size failure. Repeat the smooth case at x=−0.7 and x=0.2. Perturb a fixed batch with deterministic forward evaluations; a pass at one point does not prove every backward path correct. Preserve the original finite-difference and reused/broadcast-variable tests.

### Retained lecture checkpoints

The former deck 06 closing questions remain useful: backpropagate through `(sigmoid(wx+b)−y)²`; derive matrix-product gradient shapes; explain accumulated gradients for reused values and reverse-mode memory cost. The debugging checklist still includes initial near-uniform loss under small logits, activation/gradient scales, single-batch overfitting and fair multi-seed comparisons. Diagnose large initial loss rather than assuming one universal cause; controlled initial-logit assumptions matter.

Primary reading: [Stanford CS231n learning notes, gradient-check section](https://cs231n.github.io/neural-networks-3/). Curves and fixtures are original course calculations informed by those numerical-check principles.
