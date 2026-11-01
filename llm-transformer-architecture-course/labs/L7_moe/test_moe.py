import unittest
from math import comb

import numpy as np

from moe import (LossFreeBalancer, MoE, aux_loss_grad_step, max_violation, moe_params, route_stream,
                 skewed_batches, softmax, swiglu, switch_aux_loss)


class TestMoE(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(size=(64, 16))

    def test_all_experts_active_equals_dense_mixture(self):
        moe = MoE(16, 24, 4, k=4, rng=np.random.default_rng(1))
        y, info = moe.forward(self.x)
        probs = softmax(self.x @ moe.W_router)
        dense = sum(probs[:, [e]] * swiglu(self.x, *moe.experts[e]) for e in range(4))
        np.testing.assert_allclose(y, dense, atol=1e-10)

    def test_top_k_gates_sum_to_one_and_pick_largest_logits(self):
        moe = MoE(16, 24, 8, k=2, rng=np.random.default_rng(2))
        _, info = moe.forward(self.x)
        np.testing.assert_allclose(info["gates"].sum(axis=1), 1.0)
        logits = self.x @ moe.W_router
        self.assertTrue((np.sort(info["chosen"], 1) == np.sort(np.argsort(-logits, 1)[:, :2], 1)).all())
        self.assertEqual(info["counts"].sum(), 64 * 2)

    def test_capacity_drops_tokens(self):
        moe = MoE(16, 24, 8, k=2, rng=np.random.default_rng(3), capacity_factor=1.0)
        self.assertEqual(moe.capacity(64), 16)
        _, info = moe.forward(self.x + 3.0)          # shifted inputs: an unbalanced router
        self.assertTrue((info["counts"] <= 16).all())
        self.assertEqual(info["counts"].sum() + info["dropped"], 64 * 2)
        self.assertGreater(info["dropped"], 0)

    def test_shared_experts_see_every_token(self):
        a = MoE(16, 24, 4, k=1, rng=np.random.default_rng(4), n_shared=1)
        y, _ = a.forward(self.x)
        a.shared = []
        y0, _ = a.forward(self.x)
        rng = np.random.default_rng(4)
        np.testing.assert_allclose(y - y0, swiglu(self.x, *MoE(16, 24, 4, 1, rng, n_shared=1).shared[0]), atol=1e-10)

    def test_switch_loss_is_one_when_uniform_and_n_when_collapsed(self):
        n, T = 8, 800
        uniform = np.full((T, n), 1 / n)
        chosen = np.arange(T)[:, None] % n
        self.assertAlmostEqual(switch_aux_loss(uniform, chosen), 1.0)
        collapsed = np.zeros((T, n)); collapsed[:, 0] = 1
        self.assertAlmostEqual(switch_aux_loss(collapsed, np.zeros((T, 1), int)), n)

    def test_bias_changes_choice_not_gates(self):
        moe = MoE(16, 24, 8, k=2, rng=np.random.default_rng(5))
        bias = np.zeros(8); bias[7] = 100.0
        _, info = moe.forward(self.x, bias)
        self.assertTrue((info["chosen"] == 7).any(axis=1).all())
        logits = self.x @ moe.W_router
        np.testing.assert_allclose(info["gates"], softmax(np.take_along_axis(logits, info["chosen"], 1)))

    def test_balancing_lowers_max_violation(self):
        rng = np.random.default_rng(6)
        batches = skewed_batches(16, 200, 256, rng)
        mv = lambda infos: np.mean([max_violation(i["counts"]) for i in infos[-20:]])
        plain = route_stream(MoE(16, 24, 8, 2, np.random.default_rng(7)), batches)
        loss_free = route_stream(MoE(16, 24, 8, 2, np.random.default_rng(7)), batches, LossFreeBalancer(8, u=0.01))
        trained = MoE(16, 24, 8, 2, np.random.default_rng(7))
        losses = [aux_loss_grad_step(trained, x, lr=0.5) for x in batches]
        self.assertLess(losses[-1], losses[0])
        self.assertLess(mv(loss_free), mv(plain) / 3)
        self.assertLess(mv(route_stream(trained, batches[-20:])), mv(plain) / 3)

    def test_mixtral_and_deepseekmoe_numbers(self):
        total, active = moe_params(4096, 32, 32000, 32, 8, 128, 14336, 8, 2)
        self.assertEqual(total, 46_702_792_704)
        self.assertEqual(active, 12_879_925_248)
        self.assertEqual(comb(16, 2), 120)
        self.assertEqual(comb(64, 8), 4_426_165_368)


if __name__ == "__main__":
    unittest.main()
