import unittest

import numpy as np

import mlp
from autograd import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
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


def rel_err(a, b) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


class Autograd(unittest.TestCase):
    def test_reused_value_accumulates_gradient(self):
        x = Tensor([3.0])
        y = (x * x + x * 2.0).sum()  # dy/dx = 2x + 2 = 8
        y.backward()
        self.assertAlmostEqual(float(x.grad[0]), 8.0)

    def test_broadcast_bias_gradient_is_summed(self):
        b = Tensor(np.zeros((1, 3)))
        out = (Tensor(np.ones((5, 3))) + b).sum()
        out.backward()
        np.testing.assert_allclose(b.grad, np.full((1, 3), 5.0))

    def test_mlp_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(6, 4))
        y = rng.integers(0, 3, size=6)
        for act in ("relu", "tanh"):
            model = mlp.MLP([4, 5, 5, 3], activation=act, init="xavier", seed=1)
            for p in model.params:  # nonzero biases so the bias gradients are exercised too
                p.data += 0.1 * rng.normal(size=p.shape)
            loss = model(X).cross_entropy(y)
            loss.backward()
            for p in model.params:
                g = numerical_grad(lambda: float(model(X).cross_entropy(y).data), p.data)
                self.assertLess(rel_err(g, p.grad), 1e-5, f"{act} {p.shape}")

    def test_sigmoid_gradient(self):
        x = Tensor(np.array([[-2.0, 0.0, 1.5]]))
        out = x.sigmoid().sum()
        out.backward()
        g = numerical_grad(lambda: float((1 / (1 + np.exp(-x.data))).sum()), x.data)
        np.testing.assert_allclose(x.grad, g, rtol=1e-6)


class Capacity(unittest.TestCase):
    def test_xor_needs_a_hidden_layer(self):
        X, y = mlp.xor_data()
        linear = mlp.MLP([2, 2], init="xavier")
        mlp.train(linear, X, y, lr=0.5, steps=2000)
        self.assertLessEqual(mlp.accuracy(linear, X, y), 0.75)
        net = mlp.MLP([2, 8, 2], activation="tanh", init="xavier", seed=1)
        mlp.train(net, X, y, lr=0.5, steps=2000)
        self.assertEqual(mlp.accuracy(net, X, y), 1.0)

    def test_hidden_layer_solves_spirals(self):
        X, y = mlp.spirals()
        linear = mlp.MLP([2, 3])
        mlp.train(linear, X, y, lr=0.3, steps=1000)
        net = mlp.MLP([2, 64, 3], init="he")
        mlp.train(net, X, y, lr=0.3, steps=3000)
        self.assertLess(mlp.accuracy(linear, X, y), 0.7)
        self.assertGreater(mlp.accuracy(net, X, y), 0.95)

    def test_minibatch_sgd_also_trains(self):
        X, y = mlp.spirals()
        net = mlp.MLP([2, 64, 3], init="he")
        hist = mlp.train(net, X, y, lr=0.3, steps=3000, batch_size=32)
        self.assertGreater(mlp.accuracy(net, X, y), 0.9)
        self.assertLess(np.mean(hist[-100:]), np.mean(hist[:100]) / 4)


class Initialization(unittest.TestCase):
    def test_small_init_makes_signals_vanish(self):
        acts, grads = mlp.activation_stats("small", "relu")
        self.assertLess(acts[-1], 1e-6)
        self.assertLess(grads[0], 1e-9)

    def test_he_init_preserves_relu_signal(self):
        acts, grads = mlp.activation_stats("he", "relu")
        self.assertLess(max(acts) / min(acts), 2.0)
        self.assertLess(max(grads) / min(grads), 2.0)

    def test_xavier_suits_tanh_better_than_relu(self):
        tanh_acts, _ = mlp.activation_stats("xavier", "tanh")
        relu_acts, _ = mlp.activation_stats("xavier", "relu")
        self.assertGreater(tanh_acts[-1] / tanh_acts[0], relu_acts[-1] / relu_acts[0])
        self.assertLess(relu_acts[-1] / relu_acts[0], 0.1, "Xavier halves ReLU variance per layer or so")

    def test_init_std_formulas(self):
        self.assertAlmostEqual(mlp.init_std("he", 512, 10), np.sqrt(2 / 512))
        self.assertAlmostEqual(mlp.init_std("xavier", 300, 100), np.sqrt(2 / 400))


if __name__ == "__main__":
    unittest.main()
