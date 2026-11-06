import unittest

import numpy as np

import rnn


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


class BPTT(unittest.TestCase):
    def check(self, kind, every_step):
        rng = np.random.default_rng(1)
        net = rnn.RecurrentNet(kind, 3, 4, 2, seed=2)
        X = rng.normal(size=(6, 2, 3))
        Y = rng.normal(size=(6, 2, 2) if every_step else (2, 2))
        f = lambda: rnn.mse(net.forward(X, every_step), Y)[0]
        _, g = rnn.mse(net.forward(X, every_step), Y)
        dX = net.backward(g)
        for p, gp in zip(net.params() + [X], net.grads() + [dX]):
            np.testing.assert_allclose(gp, numerical_grad(f, p), rtol=1e-5, atol=1e-8)

    def test_rnn_gradients(self):
        for every in (False, True):
            self.check("rnn", every)

    def test_lstm_gradients(self):
        for every in (False, True):
            self.check("lstm", every)

    def test_gru_gradients(self):
        for every in (False, True):
            self.check("gru", every)

    def test_cross_entropy_gradient(self):
        rng = np.random.default_rng(0)
        logits, labels = rng.normal(size=(3, 2, 5)), rng.integers(0, 5, (3, 2))
        _, g = rnn.softmax_xent(logits, labels)
        np.testing.assert_allclose(g, numerical_grad(lambda: rnn.softmax_xent(logits, labels)[0], logits), atol=1e-8)


class Cells(unittest.TestCase):
    def test_same_weights_at_every_step(self):
        """A sequence of identical inputs from the zero state reaches the same state as stepping by hand."""
        cell = rnn.LSTMCell(2, 3, np.random.default_rng(0))
        x = np.ones((1, 2))
        s = cell.zero_state(1)
        for _ in range(4):
            s, _ = cell.step(x, s)
        net = rnn.RecurrentNet("lstm", 2, 3, 1, seed=0)
        net.forward(np.ones((4, 1, 2)))
        np.testing.assert_allclose(net.last_state[0], s[0])

    def test_lstm_forget_gate_open_keeps_the_cell(self):
        cell = rnn.LSTMCell(1, 2, np.random.default_rng(0), forget_bias=50.0)
        cell.params["b"][:2] = -50.0          # input gate closed
        c0 = np.array([[0.7, -0.3]])
        (_, c1), _ = cell.step(np.zeros((1, 1)), (np.zeros((1, 2)), c0))
        np.testing.assert_allclose(c1, c0, atol=1e-12)

    def test_clip_by_global_norm(self):
        g = [np.array([3.0]), np.array([4.0])]
        self.assertAlmostEqual(rnn.clip_by_global_norm(g, 1.0), 5.0)
        np.testing.assert_allclose(np.concatenate(g), [0.6, 0.8])


class Gradients(unittest.TestCase):
    def test_rnn_gradient_vanishes_or_explodes_with_the_recurrent_scale(self):
        self.assertLess(rnn.gradient_vs_scale(0.5), 1e-10)
        self.assertGreater(rnn.gradient_vs_scale(2.0), 100)

    def test_lstm_keeps_more_gradient_than_rnn(self):
        r, l = rnn.gradient_decay("rnn"), rnn.gradient_decay("lstm")
        self.assertGreater(l[0], 100 * r[0])


class Tasks(unittest.TestCase):
    def test_adding_problem_targets(self):
        X, y = rnn.adding_problem(5, 20, np.random.default_rng(0))
        self.assertTrue(np.all(X[:, :, 1].sum(0) == 2))
        np.testing.assert_allclose(y[:, 0], (X[:, :, 0] * X[:, :, 1]).sum(0))

    def test_lstm_solves_long_adding_where_rnn_does_not(self):
        rnn_mse, _ = rnn.train_adding("rnn", 50, steps=600)
        lstm_mse, _ = rnn.train_adding("lstm", 50, steps=600)
        self.assertGreater(rnn_mse, 0.1)
        self.assertLess(lstm_mse, 0.01)


if __name__ == "__main__":
    unittest.main()
