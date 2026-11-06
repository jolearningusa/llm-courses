"""Recurrent networks, LSTM and GRU with backpropagation through time, by hand in NumPy.

A sequence is an array of shape (T, B, D): T time steps, a batch of B sequences, D features.
Each cell maps (x_t, state_{t-1}) to state_t with the same weights at every step. The forward
pass keeps a cache per step; backward() walks the steps in reverse and adds every step's
contribution to the same weight gradients. That is backpropagation through time (BPTT): ordinary
backpropagation on the network unrolled over the sequence.

Run the demo:   python rnn.py   (about two minutes)
Run the checks: python -m unittest
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))   # numerically safe form of 1 / (1 + exp(-x))


# ----------------------------------------------------------------------------
# Cells. state is a tuple: (h,) for the RNN and GRU, (h, c) for the LSTM.
# ----------------------------------------------------------------------------

class Cell:
    def zero_state(self, B):
        return tuple(np.zeros((B, self.H)) for _ in range(self.n_state))

    def zero_grad(self):
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}


class RNNCell(Cell):
    """Elman network: h_t = tanh(x_t Wx + h_{t-1} Wh + b)."""
    n_state = 1

    def __init__(self, D, H, rng, recurrent_scale=1.0):
        self.H = H
        Wh = np.linalg.qr(rng.normal(size=(H, H)))[0] * recurrent_scale   # orthogonal, all singular values = scale
        self.params = {"Wx": rng.normal(0, 1 / np.sqrt(D), (D, H)), "Wh": Wh, "b": np.zeros(H)}
        self.zero_grad()

    def step(self, x, state):
        (h,) = state
        P = self.params
        hn = np.tanh(x @ P["Wx"] + h @ P["Wh"] + P["b"])
        return (hn,), (x, h, hn)

    def back(self, dstate, cache):
        (dh,) = dstate
        x, h, hn = cache
        P, G = self.params, self.grads
        da = dh * (1 - hn ** 2)
        G["Wx"] += x.T @ da
        G["Wh"] += h.T @ da
        G["b"] += da.sum(0)
        return da @ P["Wx"].T, (da @ P["Wh"].T,)


class LSTMCell(Cell):
    """LSTM with a forget gate.

    i, f, o = sigmoid gates; g = tanh candidate
    c_t = f * c_{t-1} + i * g        (the additive cell-state path)
    h_t = o * tanh(c_t)
    """
    n_state = 2

    def __init__(self, D, H, rng, forget_bias=1.0):
        self.H = H
        Wh = np.concatenate([np.linalg.qr(rng.normal(size=(H, H)))[0] for _ in range(4)], axis=1)
        b = np.zeros(4 * H)
        b[H:2 * H] = forget_bias          # gate order: i, f, o, g
        self.params = {"Wx": rng.normal(0, 1 / np.sqrt(D), (D, 4 * H)), "Wh": Wh, "b": b}
        self.zero_grad()

    def step(self, x, state):
        h, c = state
        P, H = self.params, self.H
        a = x @ P["Wx"] + h @ P["Wh"] + P["b"]
        i, f, o = sigmoid(a[:, :H]), sigmoid(a[:, H:2 * H]), sigmoid(a[:, 2 * H:3 * H])
        g = np.tanh(a[:, 3 * H:])
        cn = f * c + i * g
        tc = np.tanh(cn)
        hn = o * tc
        return (hn, cn), (x, h, c, i, f, o, g, tc)

    def back(self, dstate, cache):
        dh, dc = dstate
        x, h, c, i, f, o, g, tc = cache
        P, G = self.params, self.grads
        do = dh * tc
        dc = dc + dh * o * (1 - tc ** 2)
        di, df, dg = dc * g, dc * c, dc * i
        da = np.concatenate([di * i * (1 - i), df * f * (1 - f), do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
        G["Wx"] += x.T @ da
        G["Wh"] += h.T @ da
        G["b"] += da.sum(0)
        return da @ P["Wx"].T, (da @ P["Wh"].T, dc * f)


class GRUCell(Cell):
    """GRU as in Cho et al. (2014).

    z = sigmoid(x Wxz + h Whz + bz)          update gate
    r = sigmoid(x Wxr + h Whr + br)          reset gate
    n = tanh(x Wxn + (r * h) Whn + bn)       candidate
    h_t = z * h_{t-1} + (1 - z) * n
    """
    n_state = 1

    def __init__(self, D, H, rng):
        self.H = H
        q = lambda: np.linalg.qr(rng.normal(size=(H, H)))[0]
        self.params = {"Wx": rng.normal(0, 1 / np.sqrt(D), (D, 3 * H)),
                       "Wzr": np.concatenate([q(), q()], axis=1), "Wn": q(), "b": np.zeros(3 * H)}
        self.zero_grad()

    def step(self, x, state):
        (h,) = state
        P, H = self.params, self.H
        ax = x @ P["Wx"] + P["b"]
        azr = ax[:, :2 * H] + h @ P["Wzr"]
        z, r = sigmoid(azr[:, :H]), sigmoid(azr[:, H:])
        n = np.tanh(ax[:, 2 * H:] + (r * h) @ P["Wn"])
        hn = z * h + (1 - z) * n
        return (hn,), (x, h, z, r, n)

    def back(self, dstate, cache):
        (dh,) = dstate
        x, h, z, r, n = cache
        P, G = self.params, self.grads
        dz = dh * (h - n)
        dan = dh * (1 - z) * (1 - n ** 2)
        drh = dan @ P["Wn"].T
        dr = drh * h
        dazr = np.concatenate([dz * z * (1 - z), dr * r * (1 - r)], axis=1)
        dax = np.concatenate([dazr, dan], axis=1)
        G["Wx"] += x.T @ dax
        G["b"] += dax.sum(0)
        G["Wzr"] += h.T @ dazr
        G["Wn"] += (r * h).T @ dan
        dh_prev = dh * z + drh * r + dazr @ P["Wzr"].T
        return dax @ P["Wx"].T, (dh_prev,)


CELLS = {"rnn": RNNCell, "lstm": LSTMCell, "gru": GRUCell}


# ----------------------------------------------------------------------------
# A recurrent network: a cell unrolled over time plus a linear read-out
# ----------------------------------------------------------------------------

class RecurrentNet:
    """Many-to-one (read-out on the last step) or many-to-many (read-out on every step)."""

    def __init__(self, kind, D, H, d_out, seed=0, **cell_kw):
        rng = np.random.default_rng(seed)
        self.cell = CELLS[kind](D, H, rng, **cell_kw)
        self.Wo = rng.normal(0, 1 / np.sqrt(H), (H, d_out))
        self.bo = np.zeros(d_out)

    def params(self):
        return [*self.cell.params.values(), self.Wo, self.bo]

    def forward(self, X, every_step=False, state=None):
        T, B, _ = X.shape
        state = self.cell.zero_state(B) if state is None else state
        self.caches, hs = [], []
        for t in range(T):
            state, cache = self.cell.step(X[t], state)
            self.caches.append(cache)
            hs.append(state[0])
        self.hs, self.every_step, self.last_state = np.stack(hs), every_step, state
        return self.hs @ self.Wo + self.bo if every_step else hs[-1] @ self.Wo + self.bo

    def backward(self, dy):
        """dy: gradient of the loss with respect to the outputs. Returns the gradient for each input
        step and records ||dL/dh_t|| for every t in self.dh_norms."""
        self.cell.zero_grad()
        T = len(self.caches)
        if self.every_step:
            dhs = dy @ self.Wo.T
            self.gWo = np.einsum("tbh,tbo->ho", self.hs, dy)
            self.gbo = dy.sum((0, 1))
        else:
            dhs = np.zeros_like(self.hs)
            dhs[-1] = dy @ self.Wo.T
            self.gWo, self.gbo = self.hs[-1].T @ dy, dy.sum(0)
        dstate = tuple(np.zeros_like(s) for s in self.last_state)
        dX, self.dh_norms = [None] * T, np.zeros(T)
        for t in reversed(range(T)):
            dstate = (dstate[0] + dhs[t],) + dstate[1:]
            self.dh_norms[t] = np.linalg.norm(dstate[0])
            dX[t], dstate = self.cell.back(dstate, self.caches[t])
        return np.stack(dX)

    def grads(self):
        return [*self.cell.grads.values(), self.gWo, self.gbo]


def mse(y, target):
    d = y - target
    return float(np.mean(d ** 2)), 2 * d / d.size


def softmax_xent(logits, labels):
    """Mean cross-entropy over every position; logits (..., V), labels (...)."""
    z = logits - logits.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    p = np.exp(logp)
    flat = labels.reshape(-1)
    lp = logp.reshape(-1, logp.shape[-1])
    loss = -lp[np.arange(len(flat)), flat].mean()
    g = p.reshape(-1, p.shape[-1])
    g[np.arange(len(flat)), flat] -= 1
    return float(loss), (g / len(flat)).reshape(p.shape)


def clip_by_global_norm(grads, max_norm):
    """Rescale all gradients together if their joint norm exceeds max_norm (Pascanu et al., 2013)."""
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
    if max_norm is not None and norm > max_norm:
        for g in grads:
            g *= max_norm / norm
    return norm


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


# ----------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------

def adding_problem(B, T, rng):
    """Each step carries a random value in [0, 1] and a marker. Exactly two steps are marked, one in each
    half of the sequence; the target is the sum of the two marked values. Predicting the mean, 1.0,
    gives a mean squared error of 1/6 = 0.167, so a loss well below that means the network carried
    information across about T/2 steps (Hochreiter and Schmidhuber, 1997)."""
    X = np.zeros((T, B, 2))
    X[:, :, 0] = rng.random((T, B))
    a = rng.integers(0, T // 2, B)
    b = rng.integers(T // 2, T, B)
    X[a, np.arange(B), 1] = 1
    X[b, np.arange(B), 1] = 1
    y = (X[a, np.arange(B), 0] + X[b, np.arange(B), 0])[:, None]
    return X, y


def train_adding(kind, T, steps=1500, H=32, B=64, lr=1e-2, clip=1.0, seed=0, **cell_kw):
    """Train on fresh batches of the adding problem.

    Returns the test MSE on 1000 new sequences and the gradient norm (before clipping) at every step."""
    rng = np.random.default_rng(seed)
    net = RecurrentNet(kind, 2, H, 1, seed=seed, **cell_kw)
    opt = Adam(net.params(), lr)
    norms = []
    for _ in range(steps):
        X, y = adding_problem(B, T, rng)
        _, g = mse(net.forward(X), y)
        net.backward(g)
        grads = net.grads()
        norms.append(clip_by_global_norm(grads, clip))
        opt.step(grads)
    Xt, yt = adding_problem(1000, T, np.random.default_rng(10_000 + seed))
    return mse(net.forward(Xt), yt)[0], np.array(norms)


def gradient_decay(kind, T=100, H=32, seed=0, **cell_kw):
    """||dL/dh_t|| / ||dL/dh_T|| at initialization, for a loss on the last step only.

    Shows how much of the training signal from the end of a sequence reaches each earlier step."""
    rng = np.random.default_rng(seed)
    net = RecurrentNet(kind, 2, H, 1, seed=seed, **cell_kw)
    X, y = adding_problem(32, T, rng)
    _, g = mse(net.forward(X), y)
    net.backward(g)
    return net.dh_norms / net.dh_norms[-1]


def gradient_vs_scale(scale, T=50, H=32, seed=0):
    """||dL/dh_1|| / ||dL/dh_T|| for a tanh RNN whose recurrent matrix is orthogonal times `scale`.

    Backpropagating through one step multiplies the gradient by diag(1 - h_t^2) Wh^T. With the
    tanh derivative at most 1, the singular values of Wh decide whether the product of T such
    factors shrinks or grows; a linear network would scale it by exactly scale^(T-1)."""
    rng = np.random.default_rng(seed)
    X, y = adding_problem(32, T, rng)
    net = RecurrentNet("rnn", 2, H, 1, seed=seed, recurrent_scale=scale)
    _, g = mse(net.forward(X), y)
    net.backward(g)
    return float(net.dh_norms[0] / net.dh_norms[-1])


# A short text for a character-level language model. Any plain text works; this one is original.
TEXT = (
    "the network reads one character at a time and keeps a running summary in its hidden state. "
    "at every step it predicts the next character from that summary alone. "
    "the same weights are used at every position, so a pattern learned in one place is known everywhere. "
    "the gradient of the loss flows back through every step it has read, and each step multiplies it "
    "by the same recurrent matrix, which is why plain recurrent networks forget what they read long ago. "
    "gates let the network decide what to keep, what to write and what to forget. "
)


def char_lm(kind="lstm", H=64, steps=1500, T=32, B=32, lr=1e-2, seed=0):
    """Train a next-character model on TEXT. Returns (vocabulary, network, losses per step)."""
    vocab = sorted(set(TEXT))
    ids = np.array([vocab.index(ch) for ch in TEXT])
    V = len(vocab)
    rng = np.random.default_rng(seed)
    net = RecurrentNet(kind, V, H, V, seed=seed)
    opt = Adam(net.params(), lr)
    eye = np.eye(V)
    losses = []
    for _ in range(steps):
        starts = rng.integers(0, len(ids) - T - 1, B)
        seq = np.stack([ids[s:s + T + 1] for s in starts], axis=1)      # (T + 1, B)
        loss, g = softmax_xent(net.forward(eye[seq[:-1]], every_step=True), seq[1:])
        net.backward(g)
        grads = net.grads()
        clip_by_global_norm(grads, 1.0)
        opt.step(grads)
        losses.append(loss)
    return vocab, net, losses


def sample(vocab, net, prompt, n, temperature=1.0, seed=0):
    """Generate n characters after the prompt, one at a time, feeding each back in."""
    rng = np.random.default_rng(seed)
    eye = np.eye(len(vocab))
    out = list(prompt)
    logits = net.forward(eye[[vocab.index(ch) for ch in prompt]][:, None, :], every_step=True)[-1, 0]
    state = net.last_state
    for _ in range(n):
        z = logits / max(temperature, 1e-6)
        p = np.exp(z - z.max())
        p /= p.sum()
        k = int(np.argmax(p)) if temperature == 0 else int(rng.choice(len(p), p=p))
        out.append(vocab[k])
        logits = net.forward(eye[[k]][:, None, :], every_step=True, state=state)[-1, 0]
        state = net.last_state
    return "".join(out)


def demo() -> None:
    print("gradient reaching step t from a loss at step T = 100, relative to step T (at initialization)")
    for kind in ("rnn", "lstm", "gru"):
        r = gradient_decay(kind)
        print(f"  {kind:4s}: t = 90: {r[89]:.2e}   t = 75: {r[74]:.2e}   t = 50: {r[49]:.2e}   t = 1: {r[0]:.2e}")

    print("tanh RNN, T = 50, recurrent matrix = orthogonal x scale: ||dL/dh_1|| / ||dL/dh_50|| (linear: scale^49)")
    for scale in (0.5, 1.0, 1.5, 2.0, 4.0):
        print(f"  scale {scale:3.1f}: {gradient_vs_scale(scale):9.2e}   (linear {scale ** 49:9.2e})")

    print("adding problem, test MSE (predicting the mean gives 0.167), hidden 32, Adam, 1500 steps")
    for T in (10, 50, 100):
        res = {k: train_adding(k, T) for k in ("rnn", "lstm", "gru")}
        print(f"  T = {T:3d}: " + "   ".join(f"{k} {v[0]:.4f}" for k, v in res.items()))

    vocab, net, losses = char_lm()
    print(f"character LSTM on {len(TEXT)} characters, vocabulary {len(vocab)}: loss {losses[0]:.2f} -> "
          f"{np.mean(losses[-100:]):.3f} nats per character (uniform guessing: {np.log(len(vocab)):.2f})")
    print("  greedy:", repr(sample(vocab, net, "the gra", 60, temperature=0)))
    print("  T=2.0 :", repr(sample(vocab, net, "the gra", 60, temperature=2.0, seed=3)))


if __name__ == "__main__":
    demo()
