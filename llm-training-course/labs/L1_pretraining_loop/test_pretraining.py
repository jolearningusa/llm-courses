import math
import unittest

import torch

from model import Config, TinyLM, lm_loss
from optim import AdamW, Muon, newton_schulz5
from schedules import constant, warmup_cosine, wsd
from train import train


class TestSchedules(unittest.TestCase):
    def test_warmup_reaches_peak(self):
        for f in (constant, warmup_cosine, wsd):
            self.assertAlmostEqual(f(0, 100, warmup=10), 0.1)
            self.assertAlmostEqual(f(9, 100, warmup=10), 1.0)

    def test_cosine_ends_at_min(self):
        self.assertAlmostEqual(warmup_cosine(99, 100, warmup=10, min_ratio=0.1), 0.1)

    def test_wsd_is_flat_then_cools_down(self):
        self.assertEqual([wsd(t, 100, warmup=10, decay=0.2) for t in range(10, 80)], [1.0] * 70)
        self.assertAlmostEqual(wsd(99, 100, decay=0.2, shape="linear"), 0.0)
        self.assertAlmostEqual(wsd(99, 100, decay=0.2, shape="1-sqrt"), 0.0)
        # 1-sqrt drops faster at the start of the cool-down than linear does.
        self.assertLess(wsd(84, 100, decay=0.2, shape="1-sqrt"), wsd(84, 100, decay=0.2, shape="linear"))


class TestAdamW(unittest.TestCase):
    def test_matches_torch(self):
        torch.manual_seed(0)
        w0 = torch.randn(5, 7)
        mine, ref = w0.clone().requires_grad_(), w0.clone().requires_grad_()
        o1 = AdamW([mine], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        o2 = torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        target = torch.randn(5, 7)
        for _ in range(25):
            for w, o in ((mine, o1), (ref, o2)):
                o.zero_grad()
                ((w - target) ** 3).abs().sum().backward()
                o.step()
        torch.testing.assert_close(mine, ref, rtol=1e-5, atol=1e-6)

    def test_bias_correction_first_step(self):
        # After one step the update is lr * sign(g) (up to eps): m_hat = g, v_hat = g^2.
        w = torch.zeros(3, requires_grad=True)
        o = AdamW([w], lr=0.5, weight_decay=0.0)
        w.grad = torch.tensor([2.0, -3.0, 0.5])
        o.step()
        torch.testing.assert_close(w.detach(), torch.tensor([-0.5, 0.5, -0.5]))


class TestMuon(unittest.TestCase):
    def test_newton_schulz_flattens_singular_values(self):
        torch.manual_seed(0)
        u, _ = torch.linalg.qr(torch.randn(64, 64))
        v, _ = torch.linalg.qr(torch.randn(32, 32))
        s = torch.logspace(0, -2, 32)                      # condition number 100
        g = u[:, :32] @ torch.diag(s) @ v.T
        out = newton_schulz5(g)
        sv = torch.linalg.svdvals(out)
        self.assertGreater(sv.min().item(), 0.5)
        self.assertLess(sv.max().item(), 1.3)
        self.assertEqual(out.shape, g.shape)

    def test_update_rms_matches_adamw_scale(self):
        # Moonlight's 0.2 * sqrt(max(A, B)) makes the update RMS about 0.2.
        torch.manual_seed(1)
        w = torch.zeros(48, 96, requires_grad=True)
        o = Muon([w], lr=1.0, momentum=0.0, weight_decay=0.0, nesterov=False)
        w.grad = torch.randn(48, 96)
        o.step()
        rms = w.detach().pow(2).mean().sqrt().item()
        self.assertAlmostEqual(rms, 0.2, delta=0.04)


class TestModel(unittest.TestCase):
    def test_causal(self):
        torch.manual_seed(0)
        m = TinyLM(Config(vocab=13))
        x = torch.randint(13, (1, 32))
        y = x.clone()
        y[0, 20:] = (y[0, 20:] + 1) % 13
        torch.testing.assert_close(m(x)[0, :20], m(y)[0, :20])

    def test_z_loss(self):
        logits = torch.randn(4, 8, 13)
        targets = torch.randint(13, (4, 8))
        shifted = logits + 5.0                              # same softmax, larger normalizer
        ce1 = lm_loss(logits, targets)[1]
        total1, _ = lm_loss(logits, targets, 1e-2)
        total2, ce2 = lm_loss(shifted, targets, 1e-2)
        torch.testing.assert_close(ce1, ce2)
        self.assertGreater(total2.item(), total1.item())
        want = ce1 + 1e-2 * (torch.logsumexp(logits, -1) ** 2).mean()
        torch.testing.assert_close(total1, want)


class TestTraining(unittest.TestCase):
    def test_loss_falls(self):
        for run in ("adamw-wsd", "muon-wsd"):
            final, curve = train(run, seed=0, steps=120, lr=3e-3, log_every=40)
            self.assertTrue(math.isfinite(final))
            self.assertLess(final, curve[0][2])
            self.assertLess(final, math.log(13))           # better than uniform over the vocabulary


if __name__ == "__main__":
    unittest.main()
