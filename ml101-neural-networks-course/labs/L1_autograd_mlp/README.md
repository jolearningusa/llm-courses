# Lab L1: an autograd engine and multilayer perceptrons

Used with decks 05 to 07. CPU only, NumPy only.

| File | What it contains | Deck |
|---|---|---|
| `autograd.py` | A `Tensor` that records how it was computed, and `backward()`, reverse-mode differentiation in reverse topological order. Operations: `+`, `*`, `@`, `relu`, `tanh`, `sigmoid`, `sum`, `mean` and a fused softmax `cross_entropy`. | 06 |
| `mlp.py` | `MLP` with ReLU or tanh layers and `small`, `xavier` or `he` initialization; `train` (full or mini-batch SGD); XOR and three-spiral data; `activation_stats`, which measures the scale of activations and gradients at every layer of a deep network at initialization. | 05, 07 |

## Run

```bash
python mlp.py                # the numbers quoted in decks 05 and 07 (about 10 s)
python -m unittest           # 11 checks, a few seconds
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
