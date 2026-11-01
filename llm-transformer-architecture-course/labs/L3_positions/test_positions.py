"""Checks for Lab L3. Run: python -m unittest test_positions.py"""
import math
import unittest

import numpy as np

from positions import (alibi_bias, alibi_slopes, interpolate, rope, rope_frequencies, rope_wavelengths,
                       yarn_frequencies, yarn_ramp, yarn_temperature)


class RoPE(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.q, self.k = rng.normal(size=(1, 64)), rng.normal(size=(1, 64))

    def score(self, m, n, **kw):
        return (rope(self.q, np.array([m]), **kw) @ rope(self.k, np.array([n]), **kw).T).item()

    def test_score_depends_only_on_offset(self):
        # Deck 08: q_m . k_n is a function of n - m.
        for shift in (1, 17, 1000):
            self.assertAlmostEqual(self.score(5, 2), self.score(5 + shift, 2 + shift), places=10)
        self.assertNotAlmostEqual(self.score(5, 2), self.score(5, 3), places=3)

    def test_rotation_preserves_norm(self):
        x = rope(self.q, np.array([123]))
        self.assertAlmostEqual(np.linalg.norm(x), np.linalg.norm(self.q), places=10)

    def test_frequencies_match_paper(self):
        # theta_i = 10000^(-2(i-1)/d): first pair turns once per 2*pi tokens, last pair almost never.
        theta = rope_frequencies(128)
        self.assertEqual(theta[0], 1.0)
        self.assertAlmostEqual(theta[-1], 10000 ** (-126 / 128))
        self.assertGreater(rope_wavelengths(128)[-1], 50_000)   # slowest pair: ~54K tokens per turn


class ALiBi(unittest.TestCase):
    def test_slopes_for_eight_heads(self):
        # Deck 09: 1/2, 1/4, ..., 1/256.
        np.testing.assert_allclose(alibi_slopes(8), [2.0 ** -k for k in range(1, 9)])

    def test_bias_is_linear_in_distance_and_causal(self):
        b = alibi_bias(4, 8)[0]                 # head 0, slope 1/2
        expected = np.array([[0, -np.inf, -np.inf, -np.inf],
                             [-0.5, 0, -np.inf, -np.inf],
                             [-1.0, -0.5, 0, -np.inf],
                             [-1.5, -1.0, -0.5, 0]])
        np.testing.assert_array_equal(b, expected)


class ContextExtension(unittest.TestCase):
    def test_interpolation_stays_in_trained_range(self):
        # Deck 10: PI squeezes positions 0..L'-1 into [0, L).
        pos = interpolate(np.arange(32768), trained=4096, target=32768)
        self.assertLess(pos.max(), 4096)
        self.assertAlmostEqual(interpolate(np.array([20000]), 4096, 32768)[0], 2500.0)

    def test_pi_preserves_relative_property(self):
        rng = np.random.default_rng(1)
        q, k = rng.normal(size=(1, 32)), rng.normal(size=(1, 32))
        s = lambda m, n: (rope(q, np.array([m]), scale=0.25) @ rope(k, np.array([n]), scale=0.25).T).item()
        self.assertAlmostEqual(s(9000, 8990), s(10, 0), places=10)

    def test_yarn_keeps_fast_pairs_and_stretches_slow_pairs(self):
        d, L, Lp = 128, 4096, 65536
        theta, new = rope_frequencies(d), yarn_frequencies(d, L, Lp)
        gamma = yarn_ramp(d, L)
        fast, slow = gamma == 1, gamma == 0
        self.assertTrue(fast.any() and slow.any())
        np.testing.assert_allclose(new[fast], theta[fast])               # local order untouched
        np.testing.assert_allclose(new[slow], theta[slow] * L / Lp)      # same as PI

    def test_yarn_temperature(self):
        self.assertAlmostEqual(yarn_temperature(16), 0.1 * math.log(16) + 1)
        self.assertAlmostEqual(yarn_temperature(1), 1.0)


if __name__ == "__main__":
    unittest.main()
