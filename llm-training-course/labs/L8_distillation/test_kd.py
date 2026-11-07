import unittest

import numpy as np

from kd import softmax, soft_ce, soft_ce_grad, kd_loss
from modes import (GRID, teacher, student, forward_kl, reverse_kl, jsd, forward_kl_grad, reverse_kl_grad,
                   on_policy_grad, fit)


class TestTemperature(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.z, self.v = rng.normal(0, 2, size=(2, 7))
        self.z -= self.z.mean()
        self.v -= self.v.mean()

    def test_gradient_matches_finite_differences(self):
        for T in (1.0, 3.0, 10.0):
            eps = 1e-6
            fd = np.array([(soft_ce(self.z + eps * e, self.v, T) - soft_ce(self.z - eps * e, self.v, T)) / (2 * eps)
                           for e in np.eye(7)])
            self.assertTrue(np.allclose(soft_ce_grad(self.z, self.v, T), fd, atol=1e-7))

    def test_high_temperature_is_logit_matching(self):
        T = 1000.0
        g = T * T * soft_ce_grad(self.z, self.v, T)
        self.assertTrue(np.allclose(g, (self.z - self.v) / 7, atol=1e-3))

    def test_soft_targets_flatten(self):
        self.assertGreater(softmax(self.v, 1).max(), softmax(self.v, 10).max())
        self.assertTrue(np.allclose(softmax(self.v, 1e6), 1 / 7, atol=1e-5))

    def test_kd_loss_mixes_soft_and_hard(self):
        T, a = 4.0, 0.3
        hard = -np.log(softmax(self.z)[2])
        self.assertAlmostEqual(kd_loss(self.z, self.v, 2, T, a), a * T * T * soft_ce(self.z, self.v, T) + (1 - a) * hard)
        self.assertAlmostEqual(kd_loss(self.z, self.v, 2, T, 0.0), hard)


class TestModes(unittest.TestCase):
    def setUp(self):
        self.p = teacher(w=0.6)

    def test_divergences(self):
        q = student((0.0, 1.0))
        self.assertAlmostEqual(forward_kl(self.p, self.p), 0.0)
        self.assertGreater(forward_kl(self.p, q), 0)
        self.assertGreater(reverse_kl(self.p, q), 0)
        self.assertNotAlmostEqual(forward_kl(self.p, q), reverse_kl(self.p, q))

    def test_jsd_limits(self):
        q = student((0.0, 1.0))
        for b in (1e-4, 1 - 1e-4):
            scaled = jsd(self.p, q, b) / (b * (1 - b))
            target = forward_kl(self.p, q) if b < 0.5 else reverse_kl(self.p, q)
            self.assertAlmostEqual(scaled, target, delta=0.01 * target)

    def test_exact_gradients(self):
        th, eps = np.array([0.7, 0.2]), 1e-6
        for fn, grad in ((forward_kl, forward_kl_grad), (reverse_kl, reverse_kl_grad)):
            fd = [(fn(self.p, student(th + eps * e)) - fn(self.p, student(th - eps * e))) / (2 * eps) for e in np.eye(2)]
            self.assertTrue(np.allclose(grad(th, self.p), fd, atol=1e-6))

    def test_on_policy_estimate_is_unbiased(self):
        th = np.array([1.5, -0.5])
        rng = np.random.default_rng(0)
        est = np.mean([on_policy_grad(th, self.p, 64, rng) for _ in range(2000)], 0)
        self.assertTrue(np.allclose(est, reverse_kl_grad(th, self.p), rtol=0.05, atol=0.02))

    def test_forward_kl_covers_both_modes(self):
        mean = (GRID * self.p).sum()
        std = np.sqrt((GRID ** 2 * self.p).sum() - mean ** 2)
        for start in ((1.5, -0.5), (-1.5, -0.5), (0.5, 0.5)):
            mu, log_s = fit(lambda th: forward_kl_grad(th, self.p), params=start)
            self.assertLess(abs(mu - mean), 0.15)
            self.assertGreater(np.exp(log_s), std)

    def test_reverse_kl_picks_one_mode(self):
        for start, mode in (((1.5, -0.5), 3.0), ((-1.5, -0.5), -3.0)):
            mu, log_s = fit(lambda th: reverse_kl_grad(th, self.p), params=start)
            self.assertAlmostEqual(mu, mode, delta=0.02)
            self.assertAlmostEqual(np.exp(log_s), 0.8, delta=0.02)

    def test_on_policy_agrees_with_exact(self):
        rng = np.random.default_rng(3)
        mu, log_s = fit(lambda th: on_policy_grad(th, self.p, 64, rng), steps=1500)
        self.assertAlmostEqual(mu, 3.0, delta=0.1)
        self.assertAlmostEqual(np.exp(log_s), 0.8, delta=0.15)


if __name__ == "__main__":
    unittest.main()
