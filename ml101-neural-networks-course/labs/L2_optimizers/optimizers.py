"""Optimizers and learning-rate schedules, in NumPy.

Each optimizer is a small class with step(params, grads, lr) that updates a
list of arrays in place and keeps its own state (velocity, squared-gradient
averages, step count). The experiments at the bottom produce the numbers that
decks 08 and 09 quote.

Run the demo:   python optimizers.py
Run the checks: python -m unittest
"""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------

class SGD:
    """w <- w - lr * g"""

    def step(self, params, grads, lr):
        for p, g in zip(params, grads):
            p -= lr * g


class Momentum:
    """Heavy-ball momentum: v <- beta v + g;  w <- w - lr v.

    With nesterov=True the gradient is effectively taken at the look-ahead point
    (the common reformulation w <- w - lr (g + beta v_new)).
    """

    def __init__(self, beta: float = 0.9, nesterov: bool = False):
        self.beta, self.nesterov, self.v = beta, nesterov, None

    def step(self, params, grads, lr):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta
            v += g
            p -= lr * (g + self.beta * v if self.nesterov else v)


class AdaGrad:
    """Per-coordinate step lr / sqrt(sum of all past g^2)."""

    def __init__(self, eps: float = 1e-8):
        self.eps, self.s = eps, None

    def step(self, params, grads, lr):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        for p, g, s in zip(params, grads, self.s):
            s += g * g
            p -= lr * g / (np.sqrt(s) + self.eps)


class RMSProp:
    """Per-coordinate step lr / sqrt(running average of g^2)."""

    def __init__(self, rho: float = 0.9, eps: float = 1e-8):
        self.rho, self.eps, self.s = rho, eps, None

    def step(self, params, grads, lr):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        for p, g, s in zip(params, grads, self.s):
            s *= self.rho
            s += (1 - self.rho) * g * g
            p -= lr * g / (np.sqrt(s) + self.eps)


class Adam:
    """Kingma and Ba 2014, Algorithm 1, plus optional weight decay.

    m <- b1 m + (1-b1) g          first moment (momentum)
    v <- b2 v + (1-b2) g^2        second moment (RMSProp)
    m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)   bias correction
    w <- w - lr * m_hat / (sqrt(v_hat) + eps)

    weight_decay with decoupled=False adds wd * w to the gradient (L2 inside Adam);
    decoupled=True is AdamW (Loshchilov and Hutter): w <- w - lr * wd * w, outside
    the adaptive scaling.
    """

    def __init__(self, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=False, bias_correction=True):
        self.b1, self.b2, self.eps = b1, b2, eps
        self.wd, self.decoupled, self.bias_correction = weight_decay, decoupled, bias_correction
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads, lr):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if self.wd and not self.decoupled:
                g = g + self.wd * p
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.bias_correction:
                m_hat, v_hat = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            else:
                m_hat, v_hat = m, v
            if self.wd and self.decoupled:
                p -= lr * self.wd * p
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def make(name: str, **kw):
    return {"sgd": SGD, "momentum": Momentum, "adagrad": AdaGrad, "rmsprop": RMSProp, "adam": Adam}[name](**kw)


# ----------------------------------------------------------------------------
# Learning-rate schedules: functions of the step t = 0 .. total-1
# ----------------------------------------------------------------------------

def constant(t, total, peak):
    return peak


def step_decay(t, total, peak, drops=(0.5, 0.75), factor=0.1):
    """Multiply by `factor` at each fraction of training in `drops` (the ResNet recipe)."""
    return peak * factor ** sum(t >= d * total for d in drops)


def cosine(t, total, peak, final=0.0):
    return final + 0.5 * (peak - final) * (1 + math.cos(math.pi * t / total))


def warmup_cosine(t, total, peak, warmup=0.05, final_frac=0.1):
    """Linear warm-up over the first `warmup` fraction, then cosine down to final_frac * peak."""
    w = max(1, int(warmup * total))
    if t < w:
        return peak * (t + 1) / w
    return cosine(t - w, total - w, peak, final_frac * peak)


def inverse_sqrt(t, total, peak, warmup_steps=100):
    """The 2017 Transformer schedule shape: linear warm-up, then proportional to 1/sqrt(t)."""
    t = t + 1
    return peak * min(t / warmup_steps, math.sqrt(warmup_steps / t))


# ----------------------------------------------------------------------------
# Test problems
# ----------------------------------------------------------------------------

def quadratic(kappa: float = 100.0, angle: float = math.pi / 6):
    """f(w) = 1/2 w^T H w with curvatures 1 and kappa along axes rotated by `angle`.

    The rotation matters for Adam and RMSProp: their per-coordinate scaling helps most
    when the valley is aligned with the coordinate axes. Returns f, grad, start.
    """
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    H = R @ np.diag([1.0, kappa]) @ R.T
    return (lambda w: 0.5 * float(w @ H @ w)), (lambda w: H @ w), np.array([1.0, 1.0])


def rosenbrock():
    """f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, a curved valley with its minimum at (1, 1)."""
    f = lambda w: float((1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2)
    g = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2), 200 * (w[1] - w[0] ** 2)])
    return f, g, np.array([-1.5, 2.0])


def run(opt, f, grad, w0, lr, steps, schedule=constant):
    """Minimize f from w0. Returns final w and the loss before every step."""
    w = w0.astype(float).copy()
    hist = []
    for t in range(steps):
        hist.append(f(w))
        opt.step([w], [grad(w)], schedule(t, steps, lr))
    return w, hist


def steps_to(hist, tol):
    """First step whose loss is below tol (len(hist) if never)."""
    return next((i for i, h in enumerate(hist) if h < tol), len(hist))


def noisy_regression(n=2000, d=20, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    return X, X @ w + noise * rng.normal(size=n), w


def sgd_regression(lr_schedule, steps=3000, batch=8, seed=1, lr=0.02):
    """Mini-batch SGD on noisy least squares. Returns the final excess loss over the noise floor."""
    X, y, w_true = noisy_regression()
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    opt = SGD()
    for t in range(steps):
        idx = rng.integers(0, len(y), size=batch)
        g = 2 / batch * X[idx].T @ (X[idx] @ w - y[idx])
        opt.step([w], [g], lr_schedule(t, steps, lr))
    return float(np.sum((w - w_true) ** 2))


def best_lr(make_opt, f, grad, w0, steps, tol, grid):
    """The learning rate from grid that reaches tol fastest, and that number of steps."""
    best = (None, steps)
    for lr in grid:
        with np.errstate(all="ignore"):
            _, hist = run(make_opt(), f, grad, w0, lr, steps)
        n = steps_to(hist, tol) if all(np.isfinite(hist)) else steps
        if n < best[1]:
            best = (lr, n)
    return best


def best_final(make_opt, f, grad, w0, steps, grid):
    """The learning rate from grid with the lowest final loss, and that loss."""
    best = (None, math.inf)
    for lr in grid:
        with np.errstate(all="ignore"):
            _, hist = run(make_opt(), f, grad, w0, lr, steps)
        if np.isfinite(hist[-1]) and hist[-1] < best[1]:
            best = (lr, hist[-1])
    return best


GRID = [10 ** (k / 8) for k in range(-32, 9)]  # 1e-4 .. 10, eight per decade


def decay_only(decoupled: bool, steps: int = 20, lr: float = 0.1, wd: float = 0.1):
    """Run Adam with weight decay on a loss whose gradient is zero, from weights [10, 0.1].

    Only the decay acts. AdamW shrinks every weight by the same factor (1 - lr wd) per step.
    L2 inside Adam is divided by sqrt(v), so every weight moves by about lr per step whatever
    its size: small weights are wiped out and large ones barely shrink in relative terms.
    """
    w = np.array([10.0, 0.1])
    opt = Adam(weight_decay=wd, decoupled=decoupled)
    for _ in range(steps):
        opt.step([w], [np.zeros_like(w)], lr)
    return w


def demo() -> None:
    f, g, w0 = quadratic(100.0)
    print("Quadratic, condition number 100: steps to f < 1e-8 at the best learning rate")
    for name, mk in [("gd", lambda: SGD()), ("momentum 0.9", lambda: Momentum(0.9)),
                     ("nesterov 0.9", lambda: Momentum(0.9, nesterov=True)), ("adam", lambda: Adam())]:
        lr, n = best_lr(mk, f, g, w0, 3000, 1e-8, GRID)
        print(f"  {name:13s} lr {lr:.3g}  steps {n}")

    f, g, w0 = rosenbrock()
    print("Rosenbrock from (-1.5, 2), 5000 steps, best learning rate for each")
    for name, mk in [("gd", lambda: SGD()), ("momentum 0.9", lambda: Momentum(0.9)),
                     ("rmsprop", lambda: RMSProp()), ("adam", lambda: Adam())]:
        lr, loss = best_final(mk, f, g, w0, 5000, GRID)
        print(f"  {name:13s} best lr {lr:.3g}  final loss {loss:.2e}")

    print("Noisy least squares, batch 8, 3000 SGD steps, peak lr 0.02: squared error of w")
    for name, sch in [("constant", constant), ("step decay", step_decay), ("cosine", cosine),
                      ("warm-up + cosine", warmup_cosine)]:
        print(f"  {name:17s} {sgd_regression(sch):.5f}")

    print("Decay only, 20 steps from w = [10, 0.1], lr 0.1, weight decay 0.1:")
    print("  AdamW (decoupled)", np.round(decay_only(True), 4))
    print("  Adam + L2        ", np.round(decay_only(False), 4))

    print("Adam's first step with and without bias correction, g = 1 every step, lr = 0.1:")
    for bc in (True, False):
        w = np.zeros(1)
        Adam(bias_correction=bc).step([w], [np.ones(1)], 0.1)
        print(f"  bias_correction={bc!s:5s}  step size {-w[0]:.4f}")


if __name__ == "__main__":
    demo()
