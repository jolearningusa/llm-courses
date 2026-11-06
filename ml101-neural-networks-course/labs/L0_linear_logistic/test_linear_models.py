import unittest

import numpy as np

import linear_models as lm


class LeastSquares(unittest.TestCase):
    def test_closed_form_recovers_weights(self):
        X, y, w_true, b_true = lm.make_regression(n=2000, noise=0.01)
        w = lm.fit_least_squares(lm.add_bias(X), y)
        np.testing.assert_allclose(w, np.append(w_true, b_true), atol=0.01)

    def test_gradient_descent_reaches_closed_form(self):
        X, y, _, _ = lm.make_regression()
        Xb = lm.add_bias(X)
        w_gd, hist = lm.gradient_descent(lambda w: lm.mse_grad(Xb, y, w), np.zeros(4), lr=0.1, steps=500,
                                         loss_fn=lambda w: lm.mse(Xb, y, w))
        np.testing.assert_allclose(w_gd, lm.fit_least_squares(Xb, y), atol=1e-6)
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(hist, hist[1:])), "loss must not go up at this learning rate")

    def test_too_large_learning_rate_diverges(self):
        # For the MSE, GD is stable only if lr < 2 / lambda_max(2/n X^T X).
        X, y, _, _ = lm.make_regression()
        Xb = lm.add_bias(X)
        lam = np.linalg.eigvalsh(2 / len(y) * Xb.T @ Xb).max()
        _, hist = lm.gradient_descent(lambda w: lm.mse_grad(Xb, y, w), np.zeros(4), lr=2.2 / lam, steps=50,
                                      loss_fn=lambda w: lm.mse(Xb, y, w))
        self.assertGreater(hist[-1], 100 * hist[0])

    def test_sgd_gets_close(self):
        X, y, _, _ = lm.make_regression()
        Xb = lm.add_bias(X)
        w, _ = lm.sgd_linear(Xb, y, lr=0.05, epochs=30, batch_size=10)
        np.testing.assert_allclose(w, lm.fit_least_squares(Xb, y), atol=0.02)

    def test_mse_gradient_is_correct(self):
        X, y, _, _ = lm.make_regression(n=20)
        w = np.random.default_rng(1).normal(size=3)
        g = lm.numerical_grad(lambda v: lm.mse(X, y, v), w)
        self.assertLess(lm.relative_error(g, lm.mse_grad(X, y, w)), 1e-6)


class Classification(unittest.TestCase):
    def test_sigmoid_is_stable(self):
        z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
        s = lm.sigmoid(z)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertAlmostEqual(s[2], 0.5)
        np.testing.assert_allclose(s + lm.sigmoid(-z), 1.0)

    def test_softmax_is_shift_invariant(self):
        Z = np.random.default_rng(0).normal(size=(4, 5))
        np.testing.assert_allclose(lm.softmax(Z), lm.softmax(Z + 1000.0))
        np.testing.assert_allclose(lm.softmax(Z).sum(axis=1), 1.0)

    def test_two_class_softmax_is_sigmoid(self):
        z = np.linspace(-4, 4, 9)
        Z = np.stack([np.zeros_like(z), z], axis=1)
        np.testing.assert_allclose(lm.softmax(Z)[:, 1], lm.sigmoid(z))

    def test_logistic_gradient_is_correct(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 4))
        y = (rng.uniform(size=30) < 0.5).astype(float)
        w = rng.normal(size=4)
        g = lm.numerical_grad(lambda v: lm.binary_cross_entropy(X, y, v), w)
        self.assertLess(lm.relative_error(g, lm.binary_cross_entropy_grad(X, y, w)), 1e-6)

    def test_softmax_gradient_is_correct(self):
        X, y = lm.make_blobs(n_per_class=5)
        Xb = lm.add_bias(X)
        W = np.random.default_rng(3).normal(size=(3, 3))
        g = lm.numerical_grad(lambda V: lm.softmax_cross_entropy(Xb, y, V), W)
        self.assertLess(lm.relative_error(g, lm.softmax_cross_entropy_grad(Xb, y, W)), 1e-6)

    def test_softmax_regression_separates_blobs(self):
        X, y = lm.make_blobs()
        Xb = lm.add_bias(X)
        W, _ = lm.gradient_descent(lambda V: lm.softmax_cross_entropy_grad(Xb, y, V), np.zeros((3, 3)),
                                   lr=0.5, steps=300)
        self.assertGreater(lm.accuracy(Xb, y, W), 0.95)

    def test_uniform_prediction_costs_log_k(self):
        X, y = lm.make_blobs(k=4)
        self.assertAlmostEqual(lm.softmax_cross_entropy(lm.add_bias(X), y, np.zeros((3, 4))), np.log(4))


class Generalization(unittest.TestCase):
    def test_training_error_falls_with_degree(self):
        rows = lm.degree_sweep([1, 3, 5, 9, 12])
        train = [r[1] for r in rows]
        self.assertTrue(all(a >= b for a, b in zip(train, train[1:])))

    def test_validation_error_is_u_shaped(self):
        rows = lm.degree_sweep([1, 3, 5, 9, 12])
        val = [r[2] for r in rows]
        best = int(np.argmin(val))
        self.assertTrue(0 < best < len(val) - 1, val)
        self.assertGreater(val[-1], 10 * val[best])

    def test_ridge_rescues_the_high_degree_fit(self):
        plain = lm.degree_sweep([12])[0]
        ridge = lm.degree_sweep([12], l2=1e-3)[0]
        self.assertGreater(ridge[1], plain[1], "ridge must fit the training set less closely")
        self.assertLess(ridge[2], plain[2] / 10, "and generalize much better")


if __name__ == "__main__":
    unittest.main()
