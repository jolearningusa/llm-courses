"""Linear and logistic regression from scratch, in NumPy.

Everything a later neural network reuses is here in its simplest form: a model
that is a matrix product, a loss, the gradient of that loss written by hand,
gradient descent, and a validation set that tells you when the model fits noise.

Run the demo:  python linear_models.py
Run the checks: python -m unittest
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def make_regression(n: int = 200, d: int = 3, noise: float = 0.1, seed: int = 0):
    """y = X w_true + b_true + noise. Returns X (n, d), y (n,), w_true (d,), b_true."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    b_true = 0.5
    y = X @ w_true + b_true + noise * rng.normal(size=n)
    return X, y, w_true, b_true


def make_blobs(n_per_class: int = 100, k: int = 3, spread: float = 0.6, seed: int = 0):
    """k Gaussian blobs in 2-D placed on a circle. Returns X (n, 2) and integer labels y (n,)."""
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(k) / k
    centers = 2.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X = np.concatenate([c + spread * rng.normal(size=(n_per_class, 2)) for c in centers])
    y = np.repeat(np.arange(k), n_per_class)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_sine(n: int = 30, noise: float = 0.2, seed: int = 0):
    """A few noisy samples of sin(2 pi x) on [0, 1], the classic overfitting example."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, size=n))
    y = np.sin(2 * np.pi * x) + noise * rng.normal(size=n)
    return x, y


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a column of ones so the bias becomes one more weight."""
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """[1, x, x^2, ..., x^degree] for a 1-D input, with x first mapped to [-1, 1]."""
    z = 2 * x - 1
    return np.stack([z ** p for p in range(degree + 1)], axis=1)


# ----------------------------------------------------------------------------
# Linear regression
# ----------------------------------------------------------------------------

def mse(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Mean squared error (1/n) * ||X w - y||^2."""
    r = X @ w - y
    return float(r @ r / len(y))


def mse_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of mse with respect to w: (2/n) X^T (X w - y)."""
    return 2.0 / len(y) * X.T @ (X @ w - y)


def fit_least_squares(X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Closed form: minimize (1/n)||Xw - y||^2 + l2 ||w||^2.

    Setting the gradient to zero gives (X^T X + n l2 I) w = X^T y. We solve that
    system instead of inverting X^T X, which is slower and less accurate.
    """
    n, d = X.shape
    A = X.T @ X + n * l2 * np.eye(d)
    return np.linalg.solve(A, X.T @ y)


def gradient_descent(grad_fn, w0: np.ndarray, lr: float, steps: int, loss_fn=None):
    """Plain full-batch gradient descent. Returns the final weights and the loss history."""
    w = w0.copy()
    history = []
    for _ in range(steps):
        if loss_fn is not None:
            history.append(loss_fn(w))
        w = w - lr * grad_fn(w)
    return w, history


def sgd_linear(X, y, lr: float, epochs: int, batch_size: int, seed: int = 0):
    """Mini-batch SGD on the mean squared error. One epoch is one pass over shuffled data."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    history = []
    for _ in range(epochs):
        perm = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = perm[start:start + batch_size]
            w -= lr * mse_grad(X[idx], y[idx], w)
        history.append(mse(X, y, w))
    return w, history


# ----------------------------------------------------------------------------
# Logistic regression and softmax
# ----------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-z), computed without overflow for large |z|."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def binary_cross_entropy(X, y, w) -> float:
    """Mean negative log-likelihood of labels y in {0, 1} under p = sigmoid(X w).

    Written with logaddexp: -log sigmoid(z) = log(1 + e^-z), which stays finite
    even when sigmoid(z) rounds to 0 or 1.
    """
    z = X @ w
    return float(np.mean(y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)))


def binary_cross_entropy_grad(X, y, w) -> np.ndarray:
    """(1/n) X^T (sigmoid(X w) - y): the same shape as the least-squares gradient."""
    return X.T @ (sigmoid(X @ w) - y) / len(y)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row maximum changes nothing mathematically and prevents overflow."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def log_softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax_cross_entropy(X, y, W) -> float:
    """Mean of -log softmax(X W)[i, y_i]. W has shape (d, k)."""
    return float(-np.mean(log_softmax(X @ W)[np.arange(len(y)), y]))


def softmax_cross_entropy_grad(X, y, W) -> np.ndarray:
    """(1/n) X^T (P - Y): predicted probabilities minus one-hot targets."""
    P = softmax(X @ W)
    return X.T @ (P - one_hot(y, W.shape[1])) / len(y)


def accuracy(X, y, W) -> float:
    return float(np.mean(np.argmax(X @ W, axis=1) == y))


# ----------------------------------------------------------------------------
# Checking gradients and generalization
# ----------------------------------------------------------------------------

def numerical_grad(f, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences (f(w + eps e_i) - f(w - eps e_i)) / 2 eps, one coordinate at a time."""
    g = np.zeros_like(w, dtype=float)
    it = np.nditer(w, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = w[i]
        w[i] = old + eps
        fp = f(w)
        w[i] = old - eps
        fm = f(w)
        w[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))))


def degree_sweep(degrees, l2: float = 0.0, n_train: int = 15, seed: int = 0):
    """Fit polynomials of each degree to n_train noisy sine points; score on 200 fresh points.

    Returns a list of (degree, train_mse, val_mse).
    """
    x, y = make_sine(n=n_train, seed=seed)
    xv, yv = make_sine(n=200, seed=seed + 1000)
    rows = []
    for deg in degrees:
        Phi, Phiv = polynomial_features(x, deg), polynomial_features(xv, deg)
        w = fit_least_squares(Phi, y, l2=l2)
        rows.append((deg, mse(Phi, y, w), mse(Phiv, yv, w)))
    return rows


def demo() -> None:
    X, y, w_true, b_true = make_regression()
    Xb = add_bias(X)
    w_ls = fit_least_squares(Xb, y)
    w_gd, hist = gradient_descent(lambda w: mse_grad(Xb, y, w), np.zeros(Xb.shape[1]), lr=0.1, steps=200,
                                  loss_fn=lambda w: mse(Xb, y, w))
    print("true weights  ", np.round(np.append(w_true, b_true), 3))
    print("least squares ", np.round(w_ls, 3))
    print("200 GD steps  ", np.round(w_gd, 3), f"loss {hist[0]:.3f} -> {hist[-1]:.4f}")

    Xc, yc = make_blobs()
    Xcb = add_bias(Xc)
    W, _ = gradient_descent(lambda W: softmax_cross_entropy_grad(Xcb, yc, W), np.zeros((3, 3)), lr=0.5, steps=300)
    print(f"softmax regression on 3 blobs: loss {softmax_cross_entropy(Xcb, yc, W):.3f}, "
          f"accuracy {accuracy(Xcb, yc, W):.3f}")

    print("degree  train MSE  val MSE   (15 training points)")
    for deg, tr, va in degree_sweep([1, 3, 5, 9, 12]):
        print(f"{deg:6d}  {tr:9.4f}  {va:8.4f}")
    print("degree 12 with ridge l2=1e-3:", "train %.4f val %.4f" % degree_sweep([12], l2=1e-3)[0][1:])


if __name__ == "__main__":
    demo()
