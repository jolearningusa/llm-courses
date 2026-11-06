import unittest

import numpy as np

import regularization as R


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


def rel_err(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


class BatchNormLayer(unittest.TestCase):
    def test_training_output_is_standardized(self):
        x = np.random.default_rng(0).normal(3.0, 5.0, size=(256, 8))
        y = R.BatchNorm(8).forward(x, train=True)
        np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(y.std(axis=0), 1, atol=1e-4)

    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(1)
        bn = R.BatchNorm(4)
        bn.gamma[:] = rng.normal(size=4)
        bn.beta[:] = rng.normal(size=4)
        x = rng.normal(size=(6, 4))
        upstream = rng.normal(size=(6, 4))
        loss = lambda: float((bn.forward(x, train=True) * upstream).sum())
        loss()
        gx = bn.backward(upstream)
        self.assertLess(rel_err(numerical_grad(loss, x), gx), 1e-6)
        loss()
        bn.backward(upstream)
        self.assertLess(rel_err(numerical_grad(loss, bn.gamma), bn.grads[0]), 1e-6)

    def test_evaluation_uses_running_statistics(self):
        rng = np.random.default_rng(2)
        bn = R.BatchNorm(3)
        for _ in range(200):
            bn.forward(rng.normal(2.0, 3.0, size=(64, 3)), train=True)
        np.testing.assert_allclose(bn.run_mean, 2.0, atol=0.3)
        np.testing.assert_allclose(bn.run_var, 9.0, atol=1.5)
        x = rng.normal(size=(5, 3))
        alone = bn.forward(x[:1], train=False)
        in_batch = bn.forward(x, train=False)[:1]
        np.testing.assert_allclose(alone, in_batch, err_msg="in evaluation an example's output must not depend on its batch")


class DropoutLayer(unittest.TestCase):
    def test_inverted_dropout_keeps_the_expectation(self):
        d = R.Dropout(0.5, np.random.default_rng(0))
        x = np.ones((2000, 50))
        y = d.forward(x, train=True)
        self.assertAlmostEqual(float((y == 0).mean()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(y.mean()), 1.0, delta=0.01)

    def test_identity_in_evaluation(self):
        d = R.Dropout(0.5, np.random.default_rng(0))
        x = np.random.default_rng(1).normal(size=(4, 4))
        np.testing.assert_array_equal(d.forward(x, train=False), x)


class Network(unittest.TestCase):
    def test_network_with_batchnorm_gradients(self):
        rng = np.random.default_rng(3)
        net = R.Net([3, 5, 5, 2], batchnorm=True, seed=4)
        X, y = rng.normal(size=(7, 3)), rng.integers(0, 2, size=7)
        loss = lambda: R.softmax_xent(net.forward(X, train=True), y)[0]
        _, g = R.softmax_xent(net.forward(X, train=True), y)
        net.backward(g)
        for layer, p, gp in list(net.params_and_grads()):
            # A bias right before batch norm has zero gradient (the mean is subtracted), so compare
            # with an absolute tolerance as well as a relative one.
            np.testing.assert_allclose(numerical_grad(loss, p), gp, rtol=1e-5, atol=1e-8, err_msg=type(layer).__name__)

    def test_weight_decay_is_an_l2_penalty_for_sgd(self):
        Xtr, ytr, Xva, yva = R.splits()
        a, b = R.Net([2, 8, 2], seed=0), R.Net([2, 8, 2], seed=0)
        R.train(a, Xtr, ytr, Xva, yva, lr=0.1, weight_decay=0.05, epochs=1)
        # By hand: gradient of loss + (wd/2)||W||^2 on the weight matrices
        _, g = R.softmax_xent(b.forward(Xtr), ytr)
        b.backward(g)
        for layer, p, gp in list(b.params_and_grads()):
            p -= 0.1 * (gp + (0.05 * p if isinstance(layer, R.Linear) and p is layer.W else 0))
        for wa, wb in zip(a.weights(), b.weights()):
            np.testing.assert_allclose(wa, wb)


class Experiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = R.regularization_table()

    def test_unregularized_net_memorizes_and_overfits(self):
        r = self.rows["none"]
        self.assertEqual(r["train_acc"], 1.0, "it fits even the flipped labels")
        self.assertGreater(r["val_loss"], 2 * r["best_val_loss"])
        self.assertLess(r["best_epoch"], 1000, "early stopping would stop early")

    def test_weight_decay_helps(self):
        none, wd = self.rows["none"], self.rows["weight decay 1e-2"]
        self.assertLess(wd["val_loss"], none["val_loss"] / 2)
        self.assertGreater(wd["val_acc"], none["val_acc"])
        self.assertLess(wd["weight_norm"], none["weight_norm"] / 3)
        self.assertLess(wd["train_acc"], 1.0, "it no longer fits every flipped label")

    def test_dropout_with_early_stopping_helps(self):
        self.assertGreater(self.rows["dropout 0.5"]["best_val_acc"], self.rows["none"]["best_val_acc"])

    def test_batchnorm_tolerates_large_learning_rates(self):
        res = R.batchnorm_lr_sweep()
        self.assertFalse(np.isfinite(res[(False, 1.0)]) and res[(False, 1.0)] < 0.3)
        for lr in (0.03, 0.3, 1.0):
            self.assertLess(res[(True, lr)], 0.25)
        self.assertLess(res[(True, 0.03)], res[(False, 0.03)])


if __name__ == "__main__":
    unittest.main()
