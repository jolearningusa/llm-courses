"""A small reverse-mode automatic differentiation engine on NumPy arrays.

Each Tensor remembers the tensors it was computed from and a function that,
given the gradient of the loss with respect to the Tensor's value, adds the
right gradient into each parent. backward() visits the graph in reverse
topological order, which is backpropagation: the chain rule applied once per
edge, reusing every intermediate gradient.

Only the operations an MLP classifier needs are implemented.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad over the axes NumPy broadcast, so it has the parent's shape again."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, parents: tuple = (), op: str = ""):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self.op!r})"

    # -- elementwise arithmetic ------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)
        out._backward = backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else Tensor(-np.asarray(other, dtype=float)))

    __radd__ = __add__
    __rmul__ = __mul__

    # -- linear algebra ----------------------------------------------------------
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def backward():
            # Y = A B  =>  dA = dY B^T,  dB = A^T dY
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = backward
        return out

    # -- nonlinearities ----------------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def backward():
            self.grad += out.grad * (1.0 - t ** 2)
        out._backward = backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def backward():
            self.grad += out.grad * s * (1.0 - s)
        out._backward = backward
        return out

    # -- reductions --------------------------------------------------------------
    def sum(self):
        out = Tensor(self.data.sum(), (self,), "sum")

        def backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.data.size)

    def cross_entropy(self, labels: np.ndarray):
        """Mean softmax cross-entropy of logits (n, k) against integer labels (n,).

        Fused into one node: its gradient is (softmax - one_hot) / n, which is
        simpler and more stable than chaining exp, sum, log and index nodes.
        """
        n = self.data.shape[0]
        z = self.data - self.data.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        out = Tensor(-logp[np.arange(n), labels].mean(), (self,), "xent")

        def backward():
            p = np.exp(logp)
            p[np.arange(n), labels] -= 1.0
            self.grad += out.grad * p / n
        out._backward = backward
        return out

    # -- backpropagation -----------------------------------------------------------
    def backward(self):
        """Fill .grad of every tensor this one depends on. Call on a scalar loss."""
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                visit(p)
            order.append(t)
        visit(self)
        for t in order:
            t.grad = np.zeros_like(t.data)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()
        return order
