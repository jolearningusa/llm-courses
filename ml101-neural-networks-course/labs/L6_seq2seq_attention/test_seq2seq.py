import unittest

import numpy as np

import seq2seq as S
from rnn import softmax_xent


def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + eps
        fp = f()
        x[i] = old - eps
        fm = f()
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


class Gradients(unittest.TestCase):
    def check(self, attention):
        rng = np.random.default_rng(0)
        m = S.Seq2Seq(attention, d=3, H=4, A=3, seed=1)
        src, tgt_in, tgt = S.reverse_batch(2, 5, rng)
        f = lambda: softmax_xent(m.forward(src, tgt_in), tgt)[0]
        _, g = softmax_xent(m.forward(src, tgt_in), tgt)
        m.backward(g)
        for p, gp in zip(m.params(), m.grads()):
            np.testing.assert_allclose(gp, numerical_grad(f, p), rtol=1e-5, atol=1e-8)

    def test_no_attention(self):
        self.check(None)

    def test_dot_attention(self):
        self.check("dot")

    def test_additive_attention(self):
        self.check("additive")


class Mechanics(unittest.TestCase):
    def test_reverse_batch(self):
        src, tgt_in, tgt = S.reverse_batch(3, 6, np.random.default_rng(0))
        np.testing.assert_array_equal(tgt, src[::-1])
        np.testing.assert_array_equal(tgt_in[0], S.BOS)
        np.testing.assert_array_equal(tgt_in[1:], tgt[:-1])

    def test_attention_weights_are_a_distribution(self):
        m = S.Seq2Seq("additive")
        src, _, _ = S.reverse_batch(4, 7, np.random.default_rng(0))
        _, att = m.greedy(src, return_attention=True)
        self.assertEqual(att.shape, (7, 7, 4))
        np.testing.assert_allclose(att.sum(axis=1), 1.0)
        self.assertTrue(np.all(att >= 0))

    def test_greedy_matches_teacher_forcing_on_its_own_outputs(self):
        m = S.Seq2Seq("dot", seed=3)
        src, _, _ = S.reverse_batch(2, 6, np.random.default_rng(1))
        pred = m.greedy(src)
        tgt_in = np.concatenate([np.full((1, 2), S.BOS), pred[:-1]])
        np.testing.assert_array_equal(m.forward(src, tgt_in).argmax(-1), pred)


class Learning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain, _ = S.train(None, steps=800, lengths=(4, 16))
        cls.attn, _ = S.train("additive", steps=800, lengths=(4, 16))

    def test_attention_beats_a_fixed_context_on_long_inputs(self):
        p = S.accuracy_by_length(self.plain, (16,), n=200)[16][0]
        a = S.accuracy_by_length(self.attn, (16,), n=200)[16][0]
        self.assertGreater(a, p + 0.2)

    def test_attention_follows_the_anti_diagonal(self):
        src, _, _ = S.reverse_batch(50, 10, np.random.default_rng(2))
        _, att = self.attn.greedy(src, return_attention=True)
        self.assertGreater(sum(S.mirror_band(att)), 0.6, "most weight within one step of the mirrored input")


if __name__ == "__main__":
    unittest.main()
