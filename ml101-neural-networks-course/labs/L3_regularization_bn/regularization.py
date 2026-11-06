"""Regularization and batch normalization in a hand-written NumPy MLP.

Layers are small classes with forward(x, train) and backward(grad); every
backward is written out so you can read it next to deck 10 and deck 11.

Run the demo:   python regularization.py   (about a minute)
Run the checks: python -m unittest
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Data: a small, noisy training set that a big network can memorize
# ----------------------------------------------------------------------------

def two_moons(n: int, noise: float = 0.25, flip: float = 0.1, seed: int = 0):
    """Two interleaved half circles in 2-D, with a fraction `flip` of labels flipped."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, size=n)
    y = rng.integers(0, 2, size=n)
    X = np.where(y[:, None] == 0, np.stack([np.cos(t), np.sin(t)], axis=1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1))
    X += noise * rng.normal(size=X.shape)
    flipped = rng.uniform(size=n) < flip
    y = np.where(flipped, 1 - y, y)
    return X, y


def splits(n_train: int = 40, seed: int = 0):
    Xtr, ytr = two_moons(n_train, seed=seed)
    Xva, yva = two_moons(2000, seed=seed + 1, flip=0.0)
    return Xtr, ytr, Xva, yva


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------

class Linear:
    def __init__(self, d_in, d_out, rng, scale=None):
        std = np.sqrt(2.0 / d_in) if scale is None else scale
        self.W = rng.normal(0, std, size=(d_in, d_out))
        self.b = np.zeros(d_out)
        self.params = [self.W, self.b]

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.grads = [self.x.T @ g, g.sum(axis=0)]
        return g @ self.W.T


class ReLU:
    params: list = []

    def forward(self, x, train=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        self.grads = []
        return g * self.mask


class Dropout:
    """Inverted dropout: in training, zero each unit with probability p and scale the rest by 1/(1-p),
    so the expected activation is unchanged and evaluation needs no rescaling."""
    params: list = []

    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.uniform(size=x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        self.grads = []
        return g if self.mask is None else g * self.mask


class BatchNorm:
    """Ioffe and Szegedy 2015, Algorithm 1, for a (batch, features) input.

    Training: normalize with the batch mean and variance, then scale by gamma and shift by beta;
    update running averages. Evaluation: use the running averages instead of batch statistics.
    """

    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)
        self.params = [self.gamma, self.beta]
        self.run_mean, self.run_var = np.zeros(d), np.ones(d)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, train=True):
        if train:
            mu, var = x.mean(axis=0), x.var(axis=0)
            self.run_mean = (1 - self.momentum) * self.run_mean + self.momentum * mu
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var
        else:
            mu, var = self.run_mean, self.run_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.inv_std
        return self.gamma * self.xhat + self.beta

    def backward(self, g):
        n = g.shape[0]
        self.grads = [(g * self.xhat).sum(axis=0), g.sum(axis=0)]
        gx = g * self.gamma
        # The batch mean and variance depend on every row, which adds the two correction terms.
        return self.inv_std / n * (n * gx - gx.sum(axis=0) - self.xhat * (gx * self.xhat).sum(axis=0))


def softmax_xent(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    n = len(y)
    p = np.exp(logp)
    p[np.arange(n), y] -= 1
    return float(-logp[np.arange(n), y].mean()), p / n


class Net:
    def __init__(self, sizes, dropout=0.0, batchnorm=False, seed=0, init_scale=None):
        rng = np.random.default_rng(seed)
        self.layers = []
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            self.layers.append(Linear(a, b, rng, init_scale))
            if i < len(sizes) - 2:
                if batchnorm:
                    self.layers.append(BatchNorm(b))
                self.layers.append(ReLU())
                if dropout:
                    self.layers.append(Dropout(dropout, rng))

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def params_and_grads(self):
        for layer in self.layers:
            for i, p in enumerate(layer.params):
                yield layer, p, layer.grads[i]

    def weights(self):
        return [l.W for l in self.layers if isinstance(l, Linear)]


def accuracy(net, X, y):
    return float(np.mean(np.argmax(net.forward(X, train=False), axis=1) == y))


def train(net, Xtr, ytr, Xva, yva, lr=0.1, weight_decay=0.0, epochs=2000, batch=None, seed=0, eval_every=20):
    """SGD with decoupled weight decay on the weight matrices (not biases, BN gains or shifts).

    For plain SGD, decoupled decay w <- w - lr (g + wd w) is the same as adding (wd/2)||w||^2 to the loss.
    Returns a history of (epoch, train_loss, val_loss, val_acc) every eval_every epochs.
    """
    rng = np.random.default_rng(seed)
    hist = []
    for epoch in range(epochs):
        idx = np.arange(len(ytr)) if batch is None else rng.choice(len(ytr), size=batch, replace=False)
        loss, g = softmax_xent(net.forward(Xtr[idx], train=True), ytr[idx])
        net.backward(g)
        for layer, p, gp in net.params_and_grads():
            if weight_decay and isinstance(layer, Linear) and p is layer.W:
                gp = gp + weight_decay * p
            p -= lr * gp
        if epoch % eval_every == 0 or epoch == epochs - 1:
            tr, _ = softmax_xent(net.forward(Xtr, train=False), ytr)
            va, _ = softmax_xent(net.forward(Xva, train=False), yva)
            hist.append((epoch, tr, va, accuracy(net, Xva, yva)))
    return hist


def early_stopping_point(hist):
    """The history row (epoch, train_loss, val_loss, val_acc) with the lowest validation loss."""
    return min(hist, key=lambda h: h[2])


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------

BIG = [2, 256, 256, 2]


def regularization_table(epochs=4000, lr=0.3):
    """Train the same big MLP on 40 noisy points with no regularization, weight decay or dropout.

    For each: final training accuracy, final validation loss and accuracy, and the epoch with the
    lowest validation loss (where early stopping would have kept the weights) with its loss and accuracy.
    """
    Xtr, ytr, Xva, yva = splits()
    rows = {}
    for name, kw, tkw in [("none", {}, {}), ("weight decay 1e-2", {}, {"weight_decay": 1e-2}),
                          ("dropout 0.5", {"dropout": 0.5}, {})]:
        net = Net(BIG, seed=0, **kw)
        hist = train(net, Xtr, ytr, Xva, yva, lr=lr, epochs=epochs, eval_every=100, **tkw)
        best = early_stopping_point(hist)
        rows[name] = dict(train_acc=accuracy(net, Xtr, ytr), val_loss=hist[-1][2], val_acc=hist[-1][3],
                          best_epoch=best[0], best_val_loss=best[2], best_val_acc=best[3],
                          weight_norm=float(np.sqrt(sum((W ** 2).sum() for W in net.weights()))))
    return rows


def batchnorm_lr_sweep(lrs=(0.03, 0.3, 1.0), depth=8, epochs=300, seed=0):
    """Final training loss of a deep MLP with and without batch norm at several learning rates,
    on 2000 clean two-moons points with mini-batches of 64."""
    X, y = two_moons(2000, flip=0.0, seed=seed + 5)
    sizes = [2] + [64] * depth + [2]
    out = {}
    for bn in (False, True):
        for lr in lrs:
            net = Net(sizes, batchnorm=bn, seed=seed)
            with np.errstate(all="ignore"):
                hist = train(net, X, y, X[:200], y[:200], lr=lr, epochs=epochs, batch=64, eval_every=epochs)
            out[(bn, lr)] = hist[-1][1]
    return out


def demo() -> None:
    print("40 training points (10% labels flipped), 2000 clean validation points, MLP 2-256-256-2, 4000 steps")
    print(f"{'':18s} {'train acc':>9s} {'val loss':>9s} {'val acc':>8s} | {'best epoch':>10s} {'val loss':>9s} {'val acc':>8s} | {'||W||':>6s}")
    for name, r in regularization_table().items():
        print(f"{name:18s} {r['train_acc']:9.3f} {r['val_loss']:9.3f} {r['val_acc']:8.3f} | "
              f"{r['best_epoch']:10d} {r['best_val_loss']:9.3f} {r['best_val_acc']:8.3f} | {r['weight_norm']:6.1f}")
    print("\n8 hidden layers of 64, batch 64, 300 steps: final training loss")
    res = batchnorm_lr_sweep()
    for bn in (False, True):
        cells = [f"lr {lr}: " + (f"{res[(bn, lr)]:.3f}" if np.isfinite(res[(bn, lr)]) else "diverged")
                 for lr in (0.03, 0.3, 1.0)]
        print(f"  batch norm {bn!s:5s}", "  ".join(cells))


if __name__ == "__main__":
    demo()
