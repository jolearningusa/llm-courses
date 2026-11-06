"""Convolution, pooling and residual connections, by hand in NumPy.

Images are arrays of shape (N, C, H, W). conv2d is computed as a matrix product
after unfolding every receptive field into a column (im2col), which is also how
many libraries implemented it before specialized kernels.

Run the demo:   python cnn.py   (about a minute)
Run the checks: python -m unittest
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Shapes and counts
# ----------------------------------------------------------------------------

def conv_out_size(n: int, k: int, stride: int = 1, pad: int = 0) -> int:
    """Output width of a convolution: floor((n + 2 pad - k) / stride) + 1."""
    return (n + 2 * pad - k) // stride + 1


def conv_params(c_in: int, c_out: int, k: int, bias: bool = True) -> int:
    return c_out * c_in * k * k + (c_out if bias else 0)


def receptive_field(layers) -> int:
    """Receptive field of the last layer of a stack of (kernel, stride) layers, in input pixels.

    r_l = r_{l-1} + (k_l - 1) * (product of the strides of the layers before l).
    """
    r, jump = 1, 1
    for k, s in layers:
        r += (k - 1) * jump
        jump *= s
    return r


# ----------------------------------------------------------------------------
# Layers with forward and backward
# ----------------------------------------------------------------------------

def im2col(x, k, stride, pad):
    """(N, C, H, W) -> (N * Ho * Wo, C * k * k): one row per output position."""
    N, C, H, W = x.shape
    Ho, Wo = conv_out_size(H, k, stride, pad), conv_out_size(W, k, stride, pad)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.empty((N, C, k, k, Ho, Wo))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, C * k * k), (Ho, Wo)


def col2im(cols, x_shape, k, stride, pad):
    """Inverse of im2col for gradients: overlapping patches add up."""
    N, C, H, W = x_shape
    Ho, Wo = conv_out_size(H, k, stride, pad), conv_out_size(W, k, stride, pad)
    cols = cols.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(k):
        for j in range(k):
            xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += cols[:, :, i, j]
    return xp[:, :, pad:pad + H, pad:pad + W]


class Conv2d:
    def __init__(self, c_in, c_out, k, stride=1, pad=0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.k, self.stride, self.pad = k, stride, pad
        self.W = rng.normal(0, np.sqrt(2.0 / (c_in * k * k)), size=(c_out, c_in, k, k))  # He init, fan-in c_in k^2
        self.b = np.zeros(c_out)
        self.params = [self.W, self.b]

    def forward(self, x):
        self.x_shape = x.shape
        self.cols, (Ho, Wo) = im2col(x, self.k, self.stride, self.pad)
        out = self.cols @ self.W.reshape(len(self.b), -1).T + self.b
        return out.reshape(x.shape[0], Ho, Wo, -1).transpose(0, 3, 1, 2)

    def backward(self, g):
        c_out = len(self.b)
        g2 = g.transpose(0, 2, 3, 1).reshape(-1, c_out)
        self.grads = [(g2.T @ self.cols).reshape(self.W.shape), g2.sum(axis=0)]
        return col2im(g2 @ self.W.reshape(c_out, -1), self.x_shape, self.k, self.stride, self.pad)


class MaxPool2d:
    """Non-overlapping k x k max pooling. The gradient goes only to the maximum of each window."""
    params: list = []

    def __init__(self, k=2):
        self.k = k

    def forward(self, x):
        N, C, H, W = x.shape
        k = self.k
        win = x.reshape(N, C, H // k, k, W // k, k)
        out = win.max(axis=(3, 5))
        self.mask = win == out[:, :, :, None, :, None]
        self.mask = self.mask / self.mask.sum(axis=(3, 5), keepdims=True)  # split ties
        self.shape = x.shape
        return out

    def backward(self, g):
        self.grads = []
        return (self.mask * g[:, :, :, None, :, None]).reshape(self.shape)


class ReLU:
    params: list = []

    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, g):
        self.grads = []
        return g * self.m


class GlobalAvgPool:
    params: list = []

    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, g):
        self.grads = []
        N, C, H, W = self.shape
        return np.broadcast_to(g[:, :, None, None] / (H * W), self.shape).copy()


class Flatten:
    params: list = []

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, g):
        self.grads = []
        return g.reshape(self.shape)


class Linear:
    def __init__(self, d_in, d_out, rng, scale=None):
        self.W = rng.normal(0, np.sqrt(2.0 / d_in) if scale is None else scale, size=(d_in, d_out))
        self.b = np.zeros(d_out)
        self.params = [self.W, self.b]

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.grads = [self.x.T @ g, g.sum(axis=0)]
        return g @ self.W.T


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

    def params_and_grads(self):
        for l in self.layers:
            if isinstance(l, (Sequential, Residual)):
                yield from l.params_and_grads()
            else:
                for p, gp in zip(l.params, l.grads):
                    yield p, gp

    def num_params(self):
        return sum(p.size for p, _ in self._params())

    def _params(self):
        for l in self.layers:
            if isinstance(l, (Sequential, Residual)):
                yield from l._params()
            else:
                for p in l.params:
                    yield p, None


class Residual:
    """y = x + F(x): the gradient reaching x is g + (gradient through F)."""

    def __init__(self, inner: Sequential):
        self.inner = inner

    def forward(self, x):
        return x + self.inner.forward(x)

    def backward(self, g):
        return g + self.inner.backward(g)

    def params_and_grads(self):
        yield from self.inner.params_and_grads()

    def _params(self):
        yield from self.inner._params()


def softmax_xent(logits, y):
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    p = np.exp(logp)
    n = len(y)
    p[np.arange(n), y] -= 1
    return float(-logp[np.arange(n), y].mean()), p / n


def sgd_train(model, X, y, lr, steps, batch=64, seed=0, momentum=0.9):
    """SGD with momentum. Returns the training loss at every step."""
    rng = np.random.default_rng(seed)
    vel = {}
    hist = []
    for _ in range(steps):
        idx = rng.choice(len(y), size=min(batch, len(y)), replace=False)
        loss, g = softmax_xent(model.forward(X[idx]), y[idx])
        model.backward(g)
        for p, gp in model.params_and_grads():
            v = vel.setdefault(id(p), np.zeros_like(p))
            v *= momentum
            v += gp
            p -= lr * v
        hist.append(loss)
    return hist


def accuracy(model, X, y, batch=500):
    preds = np.concatenate([np.argmax(model.forward(X[i:i + batch]), axis=1) for i in range(0, len(X), batch)])
    return float(np.mean(preds == y))


# ----------------------------------------------------------------------------
# Data: small shapes at random positions
# ----------------------------------------------------------------------------

SHAPES = ["horizontal bar", "vertical bar", "plus", "square ring"]


def shapes(n: int, size: int = 12, noise: float = 0.3, region=None, seed: int = 0):
    """n single-channel images of one of four 5x5 shapes, placed at a random position.

    region=(lo, hi) limits the top-left column of the shape, which lets the checks train on
    shapes in one part of the image and test on shapes moved somewhere else.
    """
    rng = np.random.default_rng(seed)
    glyphs = np.zeros((4, 5, 5))
    glyphs[0, 2, :] = 1
    glyphs[1, :, 2] = 1
    glyphs[2, 2, :] = glyphs[2, :, 2] = 1
    glyphs[3] = 1
    glyphs[3, 1:4, 1:4] = 0
    X = noise * rng.normal(size=(n, 1, size, size))
    y = rng.integers(0, 4, size=n)
    lo, hi = region or (0, size - 5)
    for i in range(n):
        r, c = rng.integers(0, size - 4), rng.integers(lo, hi + 1)
        X[i, 0, r:r + 5, c:c + 5] += glyphs[y[i]]
    return X, y


def small_cnn(seed=0):
    rng = np.random.default_rng(seed)
    return Sequential(Conv2d(1, 8, 3, pad=1, rng=rng), ReLU(), MaxPool2d(2),
                      Conv2d(8, 16, 3, pad=1, rng=rng), ReLU(), GlobalAvgPool(), Linear(16, 4, rng))


def small_mlp(seed=0, hidden=8):
    rng = np.random.default_rng(seed)
    return Sequential(Flatten(), Linear(144, hidden, rng), ReLU(), Linear(hidden, 4, rng))


def deep_mlp(depth, residual, width=32, seed=0, d_in=2, d_out=2):
    """depth hidden blocks of Linear-ReLU (plain) or x + Linear(ReLU(Linear(x))) (residual)."""
    rng = np.random.default_rng(seed)
    layers = [Linear(d_in, width, rng)]
    for _ in range(depth):
        if residual:
            layers.append(Residual(Sequential(ReLU(), Linear(width, width, rng), ReLU(),
                                              Linear(width, width, rng, scale=np.sqrt(2.0 / width) / np.sqrt(depth)))))
        else:
            layers += [ReLU(), Linear(width, width, rng)]
    layers += [ReLU(), Linear(width, d_out, rng)]
    return Sequential(*layers)


def spirals(n_per_class=100, k=3, noise=0.2, seed=0):
    """Three interleaved spiral arms (as in lab L1)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(k):
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, n_per_class) + noise * rng.normal(size=n_per_class)
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        y.append(np.full(n_per_class, c))
    return np.concatenate(X), np.concatenate(y)


def depth_experiment(depth, residual, lr=0.01, steps=1500, seed=0):
    """Gradient norm reaching the first layer at initialization (relative to its weights), and the
    training loss at the end, for a deep plain or residual MLP on the spirals."""
    X, y = spirals()
    m = deep_mlp(depth, residual, d_out=3, seed=seed)
    gn = first_layer_grad_norm(m, X, y)
    with np.errstate(all="ignore"):
        hist = sgd_train(m, X, y, lr=lr, steps=steps)
    return gn, float(np.mean(hist[-50:])), accuracy(m, X, y)


def first_layer_grad_norm(model, X, y):
    _, g = softmax_xent(model.forward(X), y)
    model.backward(g)
    W = model.layers[0].W
    return float(np.linalg.norm(model.layers[0].grads[0]) / np.linalg.norm(W))


def demo() -> None:
    print("conv output size, 224 input, 11x11 stride 4 pad 2:", conv_out_size(224, 11, 4, 2))
    print("receptive field of three 3x3 convs:", receptive_field([(3, 1)] * 3), "| two 3x3 then a 2x2 pool then 3x3:",
          receptive_field([(3, 1), (3, 1), (2, 2), (3, 1)]))
    print("params: one 7x7 conv, 64->64:", conv_params(64, 64, 7, bias=False),
          " three 3x3 convs, 64->64:", 3 * conv_params(64, 64, 3, bias=False))

    Xtr, ytr = shapes(2000, region=(0, 3), seed=0)      # shapes only in the left part
    Xte, yte = shapes(1000, region=(4, 7), seed=1)      # shapes only in the right part
    Xsame, ysame = shapes(1000, region=(0, 3), seed=2)
    for name, model in [("CNN", small_cnn()), ("MLP", small_mlp())]:
        hist = sgd_train(model, Xtr, ytr, lr=0.02, steps=600)
        print(f"{name}: {model.num_params()} params, train loss {np.mean(hist[-50:]):.3f}, "
              f"accuracy same region {accuracy(model, Xsame, ysame):.3f}, shifted region {accuracy(model, Xte, yte):.3f}")

    print("spirals, width 32, SGD momentum 0.9, lr 0.01, 1500 steps")
    print("depth | plain: grad at init, final loss, accuracy | residual: grad at init, final loss, accuracy")
    for depth in (8, 32, 64):
        p, r = depth_experiment(depth, False), depth_experiment(depth, True)
        print(f"{depth:5d} | {p[0]:9.2e} {p[1]:7.3f} {p[2]:6.3f} | {r[0]:9.2e} {r[1]:7.3f} {r[2]:6.3f}")


if __name__ == "__main__":
    demo()
