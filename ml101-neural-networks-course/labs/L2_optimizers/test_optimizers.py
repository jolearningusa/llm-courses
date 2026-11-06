import math
import unittest

import numpy as np

import optimizers as O


class AdamDetails(unittest.TestCase):
    def test_bias_corrected_first_step_is_lr(self):
        w = np.zeros(3)
        O.Adam().step([w], [np.array([5.0, -0.01, 1e3])], 0.1)
        np.testing.assert_allclose(w, [-0.1, 0.1, -0.1], rtol=1e-6)

    def test_uncorrected_first_step_is_too_large(self):
        w = np.zeros(1)
        O.Adam(bias_correction=False).step([w], [np.ones(1)], 0.1)
        self.assertAlmostEqual(-w[0], 0.1 * 0.1 / math.sqrt(0.001), places=6)

    def test_adamw_decays_every_weight_by_the_same_factor(self):
        w = O.decay_only(decoupled=True)
        np.testing.assert_allclose(w, np.array([10.0, 0.1]) * (1 - 0.01) ** 20)

    def test_l2_inside_adam_does_not(self):
        w = O.decay_only(decoupled=False)
        shrink = w / np.array([10.0, 0.1])
        self.assertGreater(shrink[0], 2 * shrink[1], "the small weight is shrunk much harder")

    def test_adagrad_steps_shrink_like_one_over_sqrt_t(self):
        w, opt, moves = np.zeros(1), O.AdaGrad(), []
        for _ in range(16):
            before = w.copy()
            opt.step([w], [np.ones(1)], 1.0)
            moves.append(float(before[0] - w[0]))
        self.assertAlmostEqual(moves[3] / moves[15], 2.0, places=5)


class Convergence(unittest.TestCase):
    def test_gd_diverges_above_two_over_lambda_max(self):
        f, g, w0 = O.quadratic(100.0)
        with np.errstate(all="ignore"):
            _, ok = O.run(O.SGD(), f, g, w0, 0.0199, 300)
            _, bad = O.run(O.SGD(), f, g, w0, 0.0201, 300)
        self.assertLess(ok[-1], ok[0])
        self.assertGreater(bad[-1], bad[0])

    def test_momentum_beats_gd_on_an_ill_conditioned_quadratic(self):
        f, g, w0 = O.quadratic(100.0)
        _, gd = O.best_lr(lambda: O.SGD(), f, g, w0, 3000, 1e-8, O.GRID)
        _, mom = O.best_lr(lambda: O.Momentum(0.9), f, g, w0, 3000, 1e-8, O.GRID)
        _, nes = O.best_lr(lambda: O.Momentum(0.9, nesterov=True), f, g, w0, 3000, 1e-8, O.GRID)
        self.assertLess(mom, gd / 3)
        self.assertLessEqual(nes, mom)

    def test_adam_solves_rosenbrock(self):
        f, g, w0 = O.rosenbrock()
        _, adam = O.best_final(lambda: O.Adam(), f, g, w0, 5000, O.GRID)
        _, gd = O.best_final(lambda: O.SGD(), f, g, w0, 5000, O.GRID)
        self.assertLess(adam, 1e-10)
        self.assertGreater(gd, 1e-5)

    def test_decaying_the_learning_rate_beats_a_constant_one(self):
        const = O.sgd_regression(O.constant)
        for sched in (O.step_decay, O.cosine, O.warmup_cosine):
            self.assertLess(O.sgd_regression(sched), const / 3, sched.__name__)


class Schedules(unittest.TestCase):
    def test_warmup_cosine_shape(self):
        lrs = [O.warmup_cosine(t, 1000, 1.0) for t in range(1000)]
        self.assertAlmostEqual(lrs[0], 1 / 50)
        self.assertAlmostEqual(max(lrs), 1.0)
        self.assertEqual(int(np.argmax(lrs)), 49)
        self.assertAlmostEqual(lrs[-1], 0.1, places=3)
        self.assertTrue(all(a >= b for a, b in zip(lrs[49:], lrs[50:])))

    def test_step_decay(self):
        self.assertEqual([O.step_decay(t, 100, 1.0) for t in (0, 49, 50, 74, 75, 99)],
                         [1.0, 1.0, 0.1, 0.1, 0.1 ** 2, 0.1 ** 2])

    def test_inverse_sqrt_peaks_at_the_end_of_warmup(self):
        lrs = [O.inverse_sqrt(t, 10000, 1.0, warmup_steps=100) for t in range(10000)]
        self.assertEqual(int(np.argmax(lrs)), 99)
        self.assertAlmostEqual(lrs[399], 0.5)


if __name__ == "__main__":
    unittest.main()
