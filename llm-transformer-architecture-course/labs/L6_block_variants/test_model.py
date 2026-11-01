import unittest

import torch

from model import Config, FFN, RMSNorm, TinyLM, count, swiglu_hidden
from train import STOI, VOCAB, make_text, train


def cfg(**kw):
    return Config(vocab=len(VOCAB), ctx=32, d=32, n_layers=2, n_heads=4, **kw)


class TestModel(unittest.TestCase):
    def test_causal_for_every_variant(self):
        torch.manual_seed(0)
        for kw in (dict(placement="post"), dict(placement="pre", norm="rms", ffn="swiglu", n_kv_heads=1)):
            with self.subTest(**kw):
                m = TinyLM(cfg(**kw)).eval()
                x = torch.randint(len(VOCAB), (1, 16))
                y = x.clone()
                y[0, 10:] = (y[0, 10:] + 1) % len(VOCAB)
                torch.testing.assert_close(m(x)[0, :10], m(y)[0, :10])

    def test_rmsnorm_formula(self):
        n = RMSNorm(4, eps=0.0)
        x = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
        rms = (torch.tensor(50.0) / 4).sqrt()
        torch.testing.assert_close(n(x), x / rms)
        self.assertNotAlmostEqual(n(x).mean().item(), 0.0, places=3)      # no re-centering

    def test_swiglu_is_parameter_matched(self):
        d = 64
        relu, swi = count(FFN(Config(vocab=13, d=d, ffn="relu"))), count(FFN(Config(vocab=13, d=d, ffn="swiglu")))
        self.assertEqual(swiglu_hidden(d), 171)
        self.assertLess(abs(relu - swi) / relu, 0.01)

    def test_gqa_shrinks_only_the_kv_projection(self):
        full, mqa = TinyLM(cfg()), TinyLM(cfg(n_kv_heads=1))
        dh = 32 // 4
        self.assertEqual(count(full) - count(mqa), 2 * 2 * 32 * dh * (4 - 1))   # 2 layers x (K and V) x d x dh x 3 heads

    def test_data_is_correct_arithmetic(self):
        for line in make_text(50, seed=0).split(";")[:-1]:
            lhs, rhs = line.split("=")
            a, b = map(int, lhs.split("+"))
            self.assertEqual(a + b, int(rhs))
        self.assertEqual(set(make_text(50, 0)) <= set(STOI), True)

    def test_training_lowers_loss_and_is_reproducible(self):
        _, c1 = train("pre-rms", seed=0, steps=40, lr=3e-3, layers=1, eval_every=20)
        _, c2 = train("pre-rms", seed=0, steps=40, lr=3e-3, layers=1, eval_every=20)
        self.assertEqual(c1, c2)
        self.assertLess(c1[-1][1], 2.4)            # below the uniform-guess loss ln(13) = 2.56


if __name__ == "__main__":
    unittest.main()
