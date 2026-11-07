import math
import unittest

import torch

from preferences import make_world, gold_reward, sample_preferences, bt_loss, train_rm, pair_accuracy
from ppo import clipped_objective, gae, kl, ppo_step, train_policy


def world_and_rm(pairs=2000):
    phi, ref = make_world()
    w, l = sample_preferences(phi, ref, pairs, seed=1)
    return phi, ref, train_rm(phi, w, l)


class TestRewardModel(unittest.TestCase):
    def test_bradley_terry_loss(self):
        d = torch.tensor([2.0, -1.0, 0.0])
        for di in d:
            self.assertAlmostEqual(bt_loss(di, torch.tensor(0.0)).item(), math.log1p(math.exp(-di.item())), 6)
        # P(a > b) + P(b > a) = 1, so the two losses are -log p and -log(1 - p).
        p = math.exp(-bt_loss(torch.tensor(1.3), torch.tensor(0.0)).item())
        q = math.exp(-bt_loss(torch.tensor(0.0), torch.tensor(1.3)).item())
        self.assertAlmostEqual(p + q, 1.0, 6)

    def test_ranks_held_out_pairs(self):
        phi, ref, rm = world_and_rm(2000)
        a, b = sample_preferences(phi, ref, 2000, seed=2, label_noise=False)
        acc = pair_accuracy(rm, phi, a, b)
        self.assertGreater(acc, 0.85)
        _, _, small = world_and_rm(30)
        self.assertLess(pair_accuracy(small, phi, a, b), acc)


class TestPPO(unittest.TestCase):
    def test_clipped_objective(self):
        old = torch.zeros(2)
        new = torch.log(torch.tensor([1.5, 0.5])).requires_grad_(True)
        adv = torch.tensor([1.0, -1.0])
        obj = clipped_objective(new, old, adv, eps=0.2)
        self.assertAlmostEqual(obj.item(), (1.2 * 1.0 + 0.8 * -1.0) / 2, 6)
        obj.backward()  # both ratios are past the clip in the direction the advantage favours
        self.assertTrue(torch.all(new.grad == 0))

    def test_clipping_limits_the_step(self):
        phi, ref, rm = world_and_rm()
        r = rm(phi).detach()
        steps = {}
        for clip in (True, False):
            theta = torch.zeros(2, requires_grad=True)
            _, steps[clip] = ppo_step(theta, phi, ref, r, epochs=20, lr=5.0, clip=clip,
                                      generator=torch.Generator().manual_seed(0))
        self.assertLess(steps[True], 0.3)
        self.assertGreater(steps[False], 30 * steps[True])

    def test_reaches_kl_regularized_optimum(self):
        phi, ref, rm = world_and_rm()
        theta, _ = train_policy(phi, ref, rm(phi).detach(), beta=0.5)
        target = rm.w.detach() / 0.5
        self.assertLess((theta - target).norm().item() / target.norm().item(), 0.02)

    def test_overoptimization(self):
        phi, ref, rm = world_and_rm()
        proxy, gold = rm(phi).detach(), gold_reward(phi)
        rows = []
        for beta in (1.0, 0.5, 0.2, 0.1):
            _, logp = train_policy(phi, ref, proxy, beta=beta)
            p = logp.exp()
            rows.append((kl(logp, ref), (p * proxy).sum().item(), (p * gold).sum().item()))
        kls, proxies, golds = zip(*rows)
        self.assertEqual(list(kls), sorted(kls))          # a smaller beta lets the policy move further
        self.assertEqual(list(proxies), sorted(proxies))  # and the proxy reward keeps rising
        self.assertEqual(max(golds), golds[1])            # but the gold reward peaks, then falls
        self.assertLess(golds[3], golds[0])

    def test_gae_limits(self):
        r = torch.tensor([1.0, 0.0, 2.0])
        v = torch.tensor([0.5, 0.4, 1.0, 0.0])
        delta = r + v[1:] - v[:-1]
        self.assertTrue(torch.allclose(gae(r, v, lam=0.0), delta))
        returns = torch.tensor([3.0, 2.0, 2.0])
        self.assertTrue(torch.allclose(gae(r, v, lam=1.0), returns - v[:-1]))


if __name__ == "__main__":
    unittest.main()
