import random
import unittest

import torch

from chat import ASSISTANT, END, USER, VOCAB, dataset, decode, next_token_example, render
from model import Config, TinyLM, masked_loss_sum
from packing import attention_mask, pack_batch, pad_batch
from sft import train


def examples(n=12, seed=0):
    return [next_token_example(c) for c in dataset(n, seed)]


def model(seed=0):
    torch.manual_seed(seed)
    return TinyLM(Config(vocab=len(VOCAB)))


class TestTemplate(unittest.TestCase):
    def test_mask_covers_assistant_replies_only(self):
        ids, mask = render([("user", "upper ab"), ("assistant", "AB"), ("user", "first xy"), ("assistant", "x")])
        trained = decode([i for i, m in zip(ids, mask) if m and i != END])
        self.assertEqual(trained, "ABx")
        # Each reply's closing <|end|> is trained, so the model learns to stop; role markers are not.
        self.assertEqual(sum(1 for i, m in zip(ids, mask) if m and i == END), 2)
        self.assertTrue(all(m == 0 for i, m in zip(ids, mask) if i in (USER, ASSISTANT)))


class TestMaskedLoss(unittest.TestCase):
    def test_masked_tokens_get_zero_gradient(self):
        ids, tgt, msk, doc, pos = pad_batch(examples(4))
        logits = model()(ids, pos, attention_mask(doc)).detach().requires_grad_(True)
        total, _ = masked_loss_sum(logits, tgt, msk)
        total.backward()
        g = logits.grad.abs().sum(-1)
        self.assertEqual(float(g[msk == 0].max()), 0.0)
        self.assertGreater(float(g[msk == 1].min()), 0.0)


class TestPacking(unittest.TestCase):
    def test_packed_equals_unpacked(self):
        exs, m = examples(12), model()
        ids, tgt, msk, doc, pos = pad_batch(exs)
        a, na = masked_loss_sum(m(ids, pos, attention_mask(doc)), tgt, msk)
        ids, tgt, msk, doc, pos = pack_batch(exs, 128)
        b, nb = masked_loss_sum(m(ids, pos, attention_mask(doc)), tgt, msk)
        self.assertEqual(int(na), int(nb))
        self.assertAlmostEqual(float(a), float(b), places=3)
        self.assertLess(ids.size(0), len(exs))            # fewer rows: padding removed

    def test_without_block_mask_examples_leak(self):
        exs, m = examples(12), model()
        ids, tgt, msk, doc, pos = pad_batch(exs)
        a, _ = masked_loss_sum(m(ids, pos, attention_mask(doc)), tgt, msk)
        ids, tgt, msk, doc, pos = pack_batch(exs, 128)
        b, _ = masked_loss_sum(m(ids, pos, attention_mask(doc, block_diagonal=False)), tgt, msk)
        self.assertGreater(abs(float(a) - float(b)), 1e-2)

    def test_packing_keeps_every_token(self):
        exs = examples(30)
        ids, tgt, msk, doc, pos = pack_batch(exs, 128)
        self.assertEqual(int((doc != -1).sum()), sum(len(x) for x, _, _ in exs))
        self.assertEqual(int(msk.sum()), sum(sum(mm) for _, _, mm in exs))


class TestLossAggregation(unittest.TestCase):
    """Tülu 3 (arXiv:2411.15124) 4.3.2: a mean taken per micro-batch is not the batch mean."""

    def test_sum_then_divide_matches_full_batch(self):
        exs, m = examples(16, seed=3), model()
        rng = random.Random(0)
        rng.shuffle(exs)

        @torch.no_grad()
        def loss(part):
            ids, tgt, msk, doc, pos = pad_batch(part)
            return masked_loss_sum(m(ids, pos, attention_mask(doc)), tgt, msk)

        full_sum, full_n = loss(exs)
        parts = [exs[:2], exs[2:9], exs[9:]]            # uneven micro-batches
        sums = [loss(p) for p in parts]
        mean_of_means = sum(float(s / n) for s, n in sums) / len(sums)
        token_mean = sum(float(s) for s, _ in sums) / sum(float(n) for _, n in sums)
        self.assertAlmostEqual(token_mean, float(full_sum / full_n), places=4)
        self.assertNotAlmostEqual(mean_of_means, float(full_sum / full_n), places=3)


class TestTraining(unittest.TestCase):
    def test_sft_beats_untrained_model(self):
        acc, _ = train("packed", seed=0, steps=250)
        self.assertGreater(acc, 0.3)                      # an untrained model scores about 0


if __name__ == "__main__":
    unittest.main()
