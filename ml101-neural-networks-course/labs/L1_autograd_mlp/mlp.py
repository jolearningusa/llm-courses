"""Multilayer perceptrons on the autograd engine, and what initialization does to them.

Run the demo:   python mlp.py
Run the checks: python -m unittest
"""
from __future__ import annotations

import numpy as np

from autograd import Tensor


# ----------------------------------------------------------------------------
# Data: problems a linear classifier cannot solve
# ----------------------------------------------------------------------------

def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def spirals(n_per_class: int = 100, k: int = 3, noise: float = 0.2, seed: int = 0):
    """k interleaved spiral arms in 2-D, labelled by arm."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + noise * rng.normal(size=n_per_class)
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        y.append(np.full(n_per_class, c))
    X, y = np.concatenate(X), np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ----------------------------------------------------------------------------
# Initialization
# ----------------------------------------------------------------------------

def init_std(scheme: str, fan_in: int, fan_out: int) -> float:
    """Standard deviation of a zero-mean Gaussian weight matrix.

    small:  0.01, the "small random numbers" default that stalls deep nets.
    xavier: sqrt(2 / (fan_in + fan_out)), Glorot and Bengio 2010, for tanh.
    he:     sqrt(2 / fan_in), He et al. 2015, for ReLU (which zeroes half its inputs).
    """
    if scheme == "small":
        return 0.01
    if scheme == "xavier":
        return float(np.sqrt(2.0 / (fan_in + fan_out)))
    if scheme == "he":
        return float(np.sqrt(2.0 / fan_in))
    raise ValueError(scheme)


class MLP:
    """Fully connected network: [Linear -> activation] * hidden, then a Linear layer to k logits."""

    def __init__(self, sizes: list[int], activation: str = "relu", init: str = "he", seed: int = 0):
        rng = np.random.default_rng(seed)
        self.activation = activation
        self.params = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            W = Tensor(rng.normal(0.0, init_std(init, fan_in, fan_out), size=(fan_in, fan_out)))
            b = Tensor(np.zeros((1, fan_out)))
            self.params += [W, b]

    def layers(self):
        return list(zip(self.params[::2], self.params[1::2]))

    def __call__(self, X: np.ndarray, keep: list | None = None) -> Tensor:
        h = Tensor(X)
        layers = self.layers()
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = h.relu() if self.activation == "relu" else h.tanh()
                if keep is not None:
                    keep.append(h)
        return h

    def num_params(self) -> int:
        return sum(p.data.size for p in self.params)


def train(model: MLP, X, y, lr: float = 0.5, steps: int = 2000, batch_size: int | None = None, seed: int = 0):
    """Minimize cross-entropy with (mini-batch) SGD. Returns the loss at every step."""
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(steps):
        idx = slice(None) if batch_size is None else rng.choice(len(y), size=batch_size, replace=False)
        loss = model(X[idx]).cross_entropy(y[idx])
        loss.backward()
        for p in model.params:
            p.data -= lr * p.grad
        history.append(float(loss.data))
    return history


def accuracy(model: MLP, X, y) -> float:
    return float(np.mean(np.argmax(model(X).data, axis=1) == y))


# ----------------------------------------------------------------------------
# Experiments for decks 06 and 07
# ----------------------------------------------------------------------------

def activation_stats(init: str, activation: str, depth: int = 10, width: int = 256, n: int = 512, seed: int = 0):
    """For a deep network at initialization on random inputs: the standard deviation of each hidden
    layer's output (forward signal) and of the loss gradient with respect to that output (backward signal)."""
    X = np.random.default_rng(seed + 1).normal(size=(n, width))
    model = MLP([width] * (depth + 1) + [10], activation=activation, init=init, seed=seed)
    hidden: list[Tensor] = []
    loss = model(X, keep=hidden).cross_entropy(np.zeros(n, dtype=int))
    loss.backward()
    act_std = [float(h.data.std()) for h in hidden]
    grad_std = [float(h.grad.std()) for h in hidden]
    return act_std, grad_std


def demo() -> None:
    X, y = xor_data()
    linear = MLP([2, 2], init="xavier")
    train(linear, X, y, lr=0.5, steps=2000)
    net = MLP([2, 8, 2], activation="tanh", init="xavier", seed=1)
    train(net, X, y, lr=0.5, steps=2000)
    print(f"XOR: linear model accuracy {accuracy(linear, X, y):.2f}, one hidden layer {accuracy(net, X, y):.2f}")

    Xs, ys = spirals()
    for sizes in ([2, 3], [2, 64, 3], [2, 64, 64, 3]):
        m = MLP(sizes, init="he")
        hist = train(m, Xs, ys, lr=0.3, steps=3000)
        print(f"spirals {str(sizes):14s} params {m.num_params():5d}  loss {hist[-1]:.3f}  "
              f"accuracy {accuracy(m, Xs, ys):.3f}")

    print("\n10-layer net at initialization: std of activations and of their gradients at layers 1, 5, 10")
    for init, act in [("small", "tanh"), ("xavier", "tanh"), ("small", "relu"), ("xavier", "relu"), ("he", "relu")]:
        a, g = activation_stats(init, act)
        print(f"  {init:6s} {act:4s}  act {a[0]:.2e} {a[4]:.2e} {a[9]:.2e}   "
              f"grad {g[0]:.2e} {g[4]:.2e} {g[9]:.2e}")


if __name__ == "__main__":
    demo()
