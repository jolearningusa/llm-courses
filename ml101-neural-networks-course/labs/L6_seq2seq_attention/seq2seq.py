"""Sequence to sequence with and without attention, by hand in NumPy.

An encoder GRU reads the source; a decoder GRU writes the target one token at a time, starting
from the encoder's last state. Without attention, that one vector is all the decoder knows about
the source. With attention, at every output step the decoder scores every encoder state, takes a
softmax over the scores and reads their weighted average (the context vector) as well.

The task is to reverse a string of digits: the first output token depends on the last input
token and the last output token on the first, so information must cross the whole sequence.

The GRU cells come from lab L5 (../L5_rnn_lstm_gru/rnn.py).

Run the demo:   python seq2seq.py   (about three minutes)
Run the checks: python -m unittest
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "L5_rnn_lstm_gru"))
from rnn import GRUCell, Adam, softmax_xent, clip_by_global_norm  # noqa: E402

DIGITS = 10
BOS = DIGITS                 # target vocabulary: the ten digits plus a start token


def softmax(e, axis):
    z = np.exp(e - e.max(axis, keepdims=True))
    return z / z.sum(axis, keepdims=True)


class Seq2Seq:
    """attention: None (fixed context), "additive" (Bahdanau et al., 2014) or "dot" (Luong et al., 2015).

    additive score:  e_j = v . tanh(s Wa + h_j Ua)
    dot score:       e_j = s . h_j
    a = softmax(e), context c = sum_j a_j h_j, output logits = [s, c] Wo + bo
    Here the score uses the decoder state after reading the previous token, as Luong et al. do;
    Bahdanau et al. score with the state before it and feed c into the GRU.
    """

    def __init__(self, attention=None, d=16, H=64, A=32, seed=0):
        rng = np.random.default_rng(seed)
        self.attention, self.H = attention, H
        self.enc, self.dec = GRUCell(d, H, rng), GRUCell(d, H, rng)
        self.p = {"Es": rng.normal(size=(DIGITS, d)), "Et": rng.normal(size=(DIGITS + 1, d))}
        d_o = 2 * H if attention else H
        self.p["Wo"] = rng.normal(0, 1 / np.sqrt(d_o), (d_o, DIGITS))
        self.p["bo"] = np.zeros(DIGITS)
        if attention == "additive":
            self.p["Wa"] = rng.normal(0, 1 / np.sqrt(H), (H, A))
            self.p["Ua"] = rng.normal(0, 1 / np.sqrt(H), (H, A))
            self.p["va"] = rng.normal(0, 1 / np.sqrt(A), A)

    def params(self):
        return [*self.p.values(), *self.enc.params.values(), *self.dec.params.values()]

    def grads(self):
        return [*(self.g[k] for k in self.p), *self.enc.grads.values(), *self.dec.grads.values()]

    # -- forward ---------------------------------------------------------------
    def encode(self, src):
        state = self.enc.zero_state(src.shape[1])
        self.enc_caches, hs = [], []
        for t in range(len(src)):
            state, cache = self.enc.step(self.p["Es"][src[t]], state)
            self.enc_caches.append(cache)
            hs.append(state[0])
        self.src, self.hs = src, np.stack(hs)                     # (Ts, B, H)
        if self.attention == "additive":
            self.HU = self.hs @ self.p["Ua"]                       # (Ts, B, A), reused at every step
        return state

    def attend(self, s):
        """Scores, weights (Ts, B) and context (B, H) for decoder state s (B, H)."""
        if self.attention == "dot":
            Z = None
            e = np.einsum("tbh,bh->tb", self.hs, s)
        else:
            Z = np.tanh(self.HU + (s @ self.p["Wa"])[None])
            e = Z @ self.p["va"]
        a = softmax(e, axis=0)
        c = np.einsum("tb,tbh->bh", a, self.hs)
        return a, c, Z

    def dec_step(self, tok, state):
        state, cache = self.dec.step(self.p["Et"][tok], state)
        s = state[0]
        if self.attention:
            a, c, Z = self.attend(s)
            out = np.concatenate([s, c], axis=1)
        else:
            a = c = Z = None
            out = s
        return state, out @ self.p["Wo"] + self.p["bo"], (cache, tok, out, a, Z)

    def forward(self, src, tgt_in):
        """Teacher forcing: tgt_in[t] is the correct previous token. Returns logits (Tt, B, 10)."""
        state = self.encode(src)
        self.dec_caches, logits = [], []
        for t in range(len(tgt_in)):
            state, lg, cache = self.dec_step(tgt_in[t], state)
            self.dec_caches.append(cache)
            logits.append(lg)
        return np.stack(logits)

    def greedy(self, src, return_attention=False):
        """Decode as many tokens as the source has, feeding back each prediction."""
        state = self.encode(src)
        tok = np.full(src.shape[1], BOS)
        out, att = [], []
        for _ in range(len(src)):
            state, lg, cache = self.dec_step(tok, state)
            tok = lg.argmax(1)
            out.append(tok)
            att.append(cache[3])
        out = np.stack(out)
        return (out, np.stack(att)) if return_attention else out

    # -- backward --------------------------------------------------------------
    def backward(self, dlogits):
        P, H = self.p, self.H
        self.g = {k: np.zeros_like(v) for k, v in P.items()}
        self.enc.zero_grad()
        self.dec.zero_grad()
        dhs = np.zeros_like(self.hs)
        dHU = np.zeros_like(self.HU) if self.attention == "additive" else None
        ds_next = np.zeros((self.hs.shape[1], H))
        for t in reversed(range(len(self.dec_caches))):
            cache, tok, out, a, Z = self.dec_caches[t]
            do = dlogits[t]
            self.g["Wo"] += out.T @ do
            self.g["bo"] += do.sum(0)
            dout = do @ P["Wo"].T
            ds = ds_next + dout[:, :H]
            if self.attention:
                s = out[:, :H]
                dc = dout[:, H:]
                dhs += a[:, :, None] * dc[None]
                da = np.einsum("tbh,bh->tb", self.hs, dc)
                de = a * (da - (a * da).sum(0, keepdims=True))
                if self.attention == "dot":
                    ds += np.einsum("tb,tbh->bh", de, self.hs)
                    dhs += de[:, :, None] * s[None]
                else:
                    self.g["va"] += np.einsum("tba,tb->a", Z, de)
                    dpre = de[:, :, None] * P["va"] * (1 - Z ** 2)
                    dHU += dpre
                    dsW = dpre.sum(0)
                    self.g["Wa"] += s.T @ dsW
                    ds += dsW @ P["Wa"].T
            dx, (ds_next,) = self.dec.back((ds,), cache)
            np.add.at(self.g["Et"], tok, dx)
        if self.attention == "additive":
            self.g["Ua"] += np.einsum("tbh,tba->ha", self.hs, dHU)
            dhs += dHU @ P["Ua"].T
        carry = ds_next                                   # the decoder started from the last encoder state
        for t in reversed(range(len(self.enc_caches))):
            dx, (carry,) = self.enc.back((carry + dhs[t],), self.enc_caches[t])
            np.add.at(self.g["Es"], self.src[t], dx)


# ----------------------------------------------------------------------------
# Data and training
# ----------------------------------------------------------------------------

def reverse_batch(B, T, rng):
    """src: random digits (T, B); target: src reversed; decoder input: BOS then target shifted right."""
    src = rng.integers(0, DIGITS, (T, B))
    tgt = src[::-1]
    tgt_in = np.concatenate([np.full((1, B), BOS), tgt[:-1]])
    return src, tgt_in, tgt


def train(attention, steps=3000, lengths=(4, 24), B=64, lr=3e-3, seed=0, **kw):
    """Each batch has one length drawn uniformly from `lengths` (inclusive). Returns the model and losses."""
    rng = np.random.default_rng(seed)
    model = Seq2Seq(attention, seed=seed, **kw)
    opt = Adam(model.params(), lr)
    losses = []
    for _ in range(steps):
        src, tgt_in, tgt = reverse_batch(B, int(rng.integers(lengths[0], lengths[1] + 1)), rng)
        loss, g = softmax_xent(model.forward(src, tgt_in), tgt)
        model.backward(g)
        grads = model.grads()
        clip_by_global_norm(grads, 1.0)
        opt.step(grads)
        losses.append(loss)
    return model, losses


def accuracy_by_length(model, lengths, n=500, seed=1):
    """Greedy decoding. Returns {length: (token accuracy, whole-sequence accuracy)}."""
    rng = np.random.default_rng(seed)
    res = {}
    for T in lengths:
        src, _, tgt = reverse_batch(n, T, rng)
        pred = model.greedy(src)
        res[T] = (float(np.mean(pred == tgt)), float(np.mean(np.all(pred == tgt, axis=0))))
    return res


def mirror_band(att):
    """Mean attention weight on input T-1-t (the digit that output step t must write) and on input
    T-t (the digit written at step t - 1, which the decoder has just read). att: (T, T, B)."""
    T = att.shape[0]
    same = np.mean([att[t, T - 1 - t].mean() for t in range(T)])
    prev = np.mean([att[t, T - t].mean() if t > 0 else 0.0 for t in range(T)])
    return float(same), float(prev)


def attention_text(att, src):
    """A text picture of one example's attention: rows are output steps, columns input positions."""
    shades = " .:-=+*#%@"
    rows = []
    for t in range(att.shape[0]):
        rows.append("".join(shades[min(int(w * 10), 9)] for w in att[t]))
    return "\n".join(rows)


def demo() -> None:
    lengths = (4, 8, 12, 16, 20, 24)
    print("reverse a digit string; trained on lengths 4 to 24, hidden 64, 3000 Adam steps")
    print("token accuracy / whole-sequence accuracy, greedy decoding on 500 new strings per length")
    print("length       " + "".join(f"{T:>14d}" for T in lengths))
    models = {}
    for att in (None, "dot", "additive"):
        model, losses = train(att)
        models[att] = model
        res = accuracy_by_length(model, lengths)
        print(f"{str(att or 'none'):12s} " + "".join(f"   {res[T][0]:5.3f}/{res[T][1]:4.2f}" for T in lengths)
              + f"   (final loss {np.mean(losses[-100:]):.3f})")
    src, _, _ = reverse_batch(1, 12, np.random.default_rng(5))
    pred, att = models["additive"].greedy(src, return_attention=True)
    print("additive attention for input", "".join(map(str, src[:, 0])), "-> output", "".join(map(str, pred[:, 0])))
    print("(rows: output steps; columns: input positions; darker = more weight)")
    print(attention_text(att[:, :, 0], src[:, 0]))
    src, _, _ = reverse_batch(500, 12, np.random.default_rng(6))
    _, att = models["additive"].greedy(src, return_attention=True)
    band = mirror_band(att)
    print(f"over 500 strings of length 12: weight on input T-1-t (the digit being written) {band[0]:.3f}, "
          f"on input T-t (the digit just written) {band[1]:.3f}, elsewhere {1 - sum(band):.3f}")


if __name__ == "__main__":
    demo()
