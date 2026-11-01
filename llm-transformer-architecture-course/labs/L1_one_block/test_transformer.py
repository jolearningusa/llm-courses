"""Checks for Lab L1. Run: python -m unittest test_transformer.py"""
import unittest

import numpy as np

from transformer import TinyTransformer, attention, causal_mask, sinusoidal_positions


def naive_attention(q, k, v, mask):
    """One query, one key at a time: the definition, with no matrix tricks."""
    h, n, d = q.shape
    out = np.zeros_like(q)
    for a in range(h):
        for i in range(n):
            scores = [q[a, i] @ k[a, j] / np.sqrt(d) if mask[i, j] else -np.inf for j in range(k.shape[1])]
            w = np.exp(np.array(scores) - max(scores))
            w /= w.sum()
            out[a, i] = w @ v[a]
    return out


class OneBlockThreeFamilies(unittest.TestCase):
    def setUp(self):
        self.model = TinyTransformer()
        self.tokens = [3, 14, 15, 9, 26, 5, 35, 8]

    def test_attention_matches_definition(self):
        rng = np.random.default_rng(1)
        q, k, v = (rng.normal(size=(2, 5, 4)) for _ in range(3))
        mask = causal_mask(5)
        np.testing.assert_allclose(attention(q, k, v, mask), naive_attention(q, k, v, mask), rtol=1e-10)

    def test_decoder_never_sees_the_future(self):
        # Change the last three tokens: outputs at the first five positions must not move.
        changed = self.tokens[:5] + [1, 2, 4]
        a, b = self.model.decode(self.tokens), self.model.decode(changed)
        np.testing.assert_allclose(a[:5], b[:5], rtol=1e-12)
        self.assertFalse(np.allclose(a[5:], b[5:]))

    def test_encoder_sees_both_directions(self):
        # Same change: in the encoder every position moves, including the first.
        changed = self.tokens[:5] + [1, 2, 4]
        a, b = self.model.encode(self.tokens), self.model.encode(changed)
        self.assertFalse(np.allclose(a[0], b[0]))

    def test_encoder_decoder_reads_the_whole_source(self):
        target = [1, 2, 3]
        a = self.model.seq2seq(self.tokens, target)
        b = self.model.seq2seq(self.tokens[:-1] + [0], target)   # change only the last source token
        self.assertEqual(a.shape, (3, 50))
        self.assertFalse(np.allclose(a[0], b[0]))                 # even the first target position sees it

    def test_attention_is_order_blind_without_positions(self):
        # Deck 01, change 3: permuting the input of unmasked attention permutes its output.
        rng = np.random.default_rng(2)
        x = rng.normal(size=(1, 6, 4))
        perm = rng.permutation(6)
        out = attention(x, x, x)
        np.testing.assert_allclose(attention(x[:, perm], x[:, perm], x[:, perm]), out[:, perm], rtol=1e-12)

    def test_sinusoid_offsets_are_linear(self):
        # For a fixed offset k, PE(pos + k) is a rotation of PE(pos) in each (sin, cos) pair.
        pe = sinusoidal_positions(20, 8)
        k = 3
        for i in range(4):
            w = 1 / 10000 ** (2 * i / 8)
            rot = np.array([[np.cos(k * w), -np.sin(k * w)], [np.sin(k * w), np.cos(k * w)]])
            np.testing.assert_allclose(pe[k:, 2 * i:2 * i + 2], pe[:-k, 2 * i:2 * i + 2] @ rot, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
