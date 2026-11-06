import math
import unittest

from budget import (GB, activation_bytes, bubble_over_ideal, chinchilla_optimal,
                    model_state_bytes, muon_overhead, mfu, pipeline_bubble, recompute_days,
                    steps_and_examples, train_days, train_flops)


class TestBudget(unittest.TestCase):
    def test_llama3_405b_compute(self):
        # Llama 3 (arXiv:2407.21783) section 1: 3.8e25 FLOPs on 15.6T tokens.
        self.assertAlmostEqual(train_flops(405e9, 15.6e12) / 3.8e25, 1.0, delta=0.01)

    def test_chinchilla_split(self):
        # Chinchilla: 70B parameters on 1.4T tokens, 20 tokens per parameter.
        n, d = chinchilla_optimal(train_flops(70e9, 1.4e12))
        self.assertAlmostEqual(n / 70e9, 1.0, places=9)
        self.assertAlmostEqual(d / 1.4e12, 1.0, places=9)

    def test_zero_figure_1(self):
        # ZeRO (arXiv:1910.02054) Figure 1: 7.5B parameters, 64-way data parallel.
        got = [model_state_bytes(7.5e9, s, 64) / GB for s in range(4)]
        for g, want in zip(got, [120, 31.4, 16.6, 1.88]):
            self.assertAlmostEqual(g, want, delta=0.05)

    def test_critical_batch_tradeoff(self):
        s_min, e_min = 1000.0, 2_000_000.0          # B_crit = 2000
        for b in [200.0, 2000.0, 20000.0]:
            s, e = steps_and_examples(b, s_min, e_min)
            self.assertAlmostEqual((s / s_min - 1) * (e / e_min - 1), 1.0)
        s, e = steps_and_examples(2000.0, s_min, e_min)
        self.assertEqual((s, e), (2 * s_min, 2 * e_min))   # twice both minimums at B_crit

    def test_muon_overhead(self):
        # Jordan (2024): about 0.7% for the NanoGPT speedrun, about 0.5% for Llama 405B.
        self.assertAlmostEqual(muon_overhead(768, 524288), 0.007, delta=0.0005)
        self.assertAlmostEqual(muon_overhead(16384, 16 * 2**20), 0.005, delta=0.0005)

    def test_mfu_and_days_are_inverse(self):
        c = train_flops(8e9, 1e12)
        days = train_days(c, 64, 1e15, 0.4)
        tps = 1e12 / (days * 86400)
        self.assertAlmostEqual(mfu(tps, 8e9, 64, 1e15), 0.4)

    def test_pipeline_bubble(self):
        # GPipe section 2.3: negligible once M >= 4K; at M = 4K the bubble is under 20% of the step.
        self.assertLess(pipeline_bubble(8, 32), 0.2)
        # Both forms agree: idle / total = x / (1 + x) with x = bubble / ideal.
        x = bubble_over_ideal(8, 32)
        self.assertAlmostEqual(pipeline_bubble(8, 32), x / (1 + x))
        # Narayanan et al. section 2.2.2: v chunks per device divide the bubble by v.
        self.assertAlmostEqual(bubble_over_ideal(8, 32, chunks=4), x / 4)

    def test_megatron_training_days(self):
        # Narayanan et al. (arXiv:2104.04473) section 5.1: GPT-3 175B on 300B tokens, 1024 GPUs at
        # 140 TFLOP/s, about 34 days; 1T parameters on 450B tokens, 3072 GPUs at 163 TFLOP/s, 84 days.
        self.assertAlmostEqual(recompute_days(175e9, 300e9, 1024, 140e12), 34, delta=0.5)
        self.assertAlmostEqual(recompute_days(1008e9, 450e9, 3072, 163e12), 84, delta=0.5)

    def test_activation_memory(self):
        # Korthikanti et al. (arXiv:2205.05198): eq. 2 at t = 1 is eq. 1, sbh(34 + 5as/h).
        s, b, h, a = 2048, 1, 12288, 96
        self.assertAlmostEqual(activation_bytes(s, b, h, a), s * b * h * (34 + 5 * a * s / h))
        # Eq. 4: sequence parallelism divides all of eq. 1 by t; Table 2: selective leaves 34sbh/t.
        full = activation_bytes(s, b, h, a)
        self.assertAlmostEqual(activation_bytes(s, b, h, a, 8, sequence_parallel=True), full / 8)
        self.assertAlmostEqual(activation_bytes(s, b, h, a, 8, True, True), 34 * s * b * h / 8)
        self.assertGreater(activation_bytes(s, b, h, a, 8), full / 8)   # tensor parallel alone

    def test_bad_stage(self):
        with self.assertRaises(ValueError):
            model_state_bytes(1e9, 4, 8)


if __name__ == "__main__":
    unittest.main()
