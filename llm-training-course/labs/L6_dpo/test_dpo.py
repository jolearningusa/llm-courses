import unittest

import numpy as np

from losses import dpo, ipo, simpo, implicit_reward
from bandit import bradley_terry, log_softmax, rlhf_optimum, train, expected_grad


def finite_diff(fn, args, k, eps=1e-6):
    a = [np.array(x, dtype=float) for x in args]
    up = [x.copy() for x in a]
    dn = [x.copy() for x in a]
    up[k] += eps
    dn[k] -= eps
    return (fn(*up)[0] - fn(*dn)[0]) / (2 * eps)


class TestLosses(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.lw, self.ll, self.rw, self.rl = rng.normal(-20, 5, size=(4, 8))
        self.nw, self.nl = rng.integers(5, 60, size=(2, 8)).astype(float)

    def test_dpo_value_and_gradients(self):
        loss, gw, gl = dpo(self.lw, self.ll, self.rw, self.rl, beta=0.1)
        z = 0.1 * ((self.lw - self.rw) - (self.ll - self.rl))
        self.assertTrue(np.allclose(loss, -np.log(1 / (1 + np.exp(-z)))))
        f = lambda lw, ll: dpo(lw, ll, self.rw, self.rl, beta=0.1)
        self.assertTrue(np.allclose(gw, finite_diff(f, (self.lw, self.ll), 0), atol=1e-6))
        self.assertTrue(np.allclose(gl, finite_diff(f, (self.lw, self.ll), 1), atol=1e-6))

    def test_dpo_at_the_reference_is_log_2(self):
        loss, gw, gl = dpo(self.rw, self.rl, self.rw, self.rl, beta=0.1)
        self.assertTrue(np.allclose(loss, np.log(2)))
        self.assertTrue(np.allclose(gw, -0.05))   # beta * sigmoid(0)

    def test_ipo_and_simpo_gradients(self):
        f = lambda lw, ll: ipo(lw, ll, self.rw, self.rl, tau=0.5)
        _, gw, gl = f(self.lw, self.ll)
        self.assertTrue(np.allclose(gw, finite_diff(f, (self.lw, self.ll), 0), atol=1e-4))
        self.assertTrue(np.allclose(gl, finite_diff(f, (self.lw, self.ll), 1), atol=1e-4))
        g = lambda lw, ll: simpo(lw, ll, self.nw, self.nl, beta=2.0, gamma=1.0)
        _, gw, gl = g(self.lw, self.ll)
        self.assertTrue(np.allclose(gw, finite_diff(g, (self.lw, self.ll), 0), atol=1e-6))
        self.assertTrue(np.allclose(gl, finite_diff(g, (self.lw, self.ll), 1), atol=1e-6))

    def test_simpo_normalizes_length(self):
        # Repeating both answers k times multiplies log-probs and lengths by k: SimPO is unchanged,
        # a summed-log-prob margin is not.
        a = simpo(self.lw, self.ll, self.nw, self.nl)[0]
        b = simpo(3 * self.lw, 3 * self.ll, 3 * self.nw, 3 * self.nl)[0]
        self.assertTrue(np.allclose(a, b))
        self.assertFalse(np.allclose(dpo(self.lw, self.ll, 0, 0)[0], dpo(3 * self.lw, 3 * self.ll, 0, 0)[0]))


class TestBandit(unittest.TestCase):
    def test_policy_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(2)
        theta, ref = rng.normal(size=5), log_softmax(rng.normal(size=5))
        P = bradley_terry(rng.normal(size=5))
        _, g = expected_grad(theta, ref, P, dpo, beta=0.3)
        for k in range(5):
            e = np.zeros(5)
            e[k] = 1e-6
            fd = (expected_grad(theta + e, ref, P, dpo, beta=0.3)[0] -
                  expected_grad(theta - e, ref, P, dpo, beta=0.3)[0]) / 2e-6
            self.assertAlmostEqual(g[k], fd, places=6)

    def test_dpo_optimum_is_the_rlhf_policy(self):
        rng = np.random.default_rng(0)
        ref, r = log_softmax(rng.normal(size=6)), rng.normal(size=6)
        for beta in (0.2, 0.5, 2.0):
            pol = train(ref, bradley_terry(r), dpo, lr=1 / beta ** 2, beta=beta)
            self.assertLess(np.abs(pol - rlhf_optimum(ref, r, beta)).max(), 1e-6)
            # The implicit reward recovers r up to one constant.
            d = implicit_reward(pol, ref, beta) - r
            self.assertLess(d.max() - d.min(), 1e-5)

    def test_deterministic_preferences(self):
        P = np.array([[0.0, 1.0], [0.0, 0.0]])
        ref = np.log(np.array([0.5, 0.5]))
        for c in (0.1, 1.0):
            d1 = np.exp(train(ref, P, dpo, steps=1000, beta=c))[1]
            d2 = np.exp(train(ref, P, dpo, steps=10000, beta=c))[1]
            self.assertLess(d2, d1 / 5)                    # DPO keeps pushing the loser to zero
            i = np.exp(train(ref, P, ipo, steps=2000, lr=0.1, tau=c))[1]
            self.assertAlmostEqual(i, 1 / (1 + np.exp(1 / (2 * c))), places=6)  # IPO stops at sigmoid(-1/(2 tau))


if __name__ == "__main__":
    unittest.main()
