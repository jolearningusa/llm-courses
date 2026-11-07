import unittest

import numpy as np

from lora import LoRALinear, best_rank_r_loss, fit, lora_params, low_rank_target
from nf4 import (PAPER_NF4, bits_per_parameter, dequantize, double_quantize, int4_codebook, mse,
                 nf4_codebook, quantize)


class TestLoRA(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.layer = LoRALinear(self.rng.standard_normal((6, 5)), r=2, alpha=4, rng=self.rng)
        self.x = self.rng.standard_normal((3, 5))

    def test_zero_init_leaves_base_model_unchanged(self):
        np.testing.assert_allclose(self.layer.forward(self.x), self.x @ self.layer.w0.T)

    def test_merged_equals_unmerged(self):
        self.layer.B = self.rng.standard_normal(self.layer.B.shape)
        before = self.layer.forward(self.x)
        self.layer.merge()
        np.testing.assert_allclose(self.layer.forward(self.x), before, atol=1e-12)

    def test_gradients_match_finite_differences(self):
        self.layer.B = self.rng.standard_normal(self.layer.B.shape)
        t = self.rng.standard_normal((3, 6))

        def loss():
            return 0.5 * float(((self.layer.forward(self.x) - t) ** 2).sum())
        ga, gb = self.layer.backward(self.x, self.layer.forward(self.x) - t)
        for p, g in ((self.layer.A, ga), (self.layer.B, gb)):
            i = (1, 1)
            old = p[i]
            p[i] = old + 1e-6
            up = loss()
            p[i] = old - 1e-6
            down = loss()
            p[i] = old
            self.assertAlmostEqual((up - down) / 2e-6, g[i], places=5)

    def test_parameter_counts_match_the_paper(self):
        # LoRA (arXiv:2106.09685) Table 4: GPT-3 175B (96 layers, width 12,288), Wq and Wv adapted.
        per_layer = lambda r: 2 * lora_params(12288, 12288, r)
        self.assertAlmostEqual(96 * per_layer(1) / 1e6, 4.7, places=1)
        self.assertAlmostEqual(96 * per_layer(8) / 1e6, 37.7, places=1)
        self.assertEqual(self.layer.trainable(), lora_params(6, 5, 2))

    def test_fit_reaches_the_best_rank_r_error(self):
        delta = low_rank_target(32, 24, rank=4)
        for r in (1, 2):
            self.assertAlmostEqual(fit(delta, r, steps=2000) / best_rank_r_loss(delta, r), 1.0, delta=0.01)
        self.assertLess(fit(delta, 4, steps=2000), 1e-3)


class TestNF4(unittest.TestCase):
    def test_codebook_matches_appendix_e(self):
        v = nf4_codebook()
        self.assertEqual(len(v), 16)
        self.assertIn(0.0, v)                                   # zero is exact: padding, ReLU zeros
        np.testing.assert_allclose(v, PAPER_NF4, atol=1e-6)

    def test_nf4_beats_int4_on_normal_weights_only(self):
        rng = np.random.default_rng(0)

        def err(w, cb):
            i, a = quantize(w, cb)
            return mse(dequantize(i, a, cb, w.shape), w)
        normal = rng.standard_normal(64 * 512)
        uniform = rng.uniform(-1, 1, 64 * 512)
        self.assertLess(err(normal, nf4_codebook()), err(normal, int4_codebook()))
        self.assertGreater(err(uniform, nf4_codebook()), err(uniform, int4_codebook()))

    def test_double_quantization_bits(self):
        # QLoRA section 3: 32/64 = 0.5 bits per parameter, 0.127 with double quantization.
        self.assertEqual(bits_per_parameter(), 0.5)
        self.assertAlmostEqual(bits_per_parameter(double=True), 0.127, places=3)

    def test_double_quantization_costs_little_accuracy(self):
        rng = np.random.default_rng(1)
        w = rng.standard_normal(64 * 256 * 4)
        cb = nf4_codebook()
        i, a = quantize(w, cb)
        a2, codes = double_quantize(a)
        self.assertEqual(codes.dtype, np.int8)
        e1, e2 = mse(dequantize(i, a, cb, w.shape), w), mse(dequantize(i, a2, cb, w.shape), w)
        self.assertLess(abs(e2 - e1) / e1, 0.01)


if __name__ == "__main__":
    unittest.main()
