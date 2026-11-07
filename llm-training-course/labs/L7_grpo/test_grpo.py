import unittest

import torch

from task import prompts, split, reward, target, decode, EOS, FILLER, MAX_LEN
from policy import Policy
from grpo import group_advantages, token_weights, policy_loss, sft, evaluate, train


class TestReward(unittest.TestCase):
    def test_rule_based_reward(self):
        self.assertEqual(reward((7, 8), target((7, 8), n_filler=2)), 1.0)
        self.assertEqual(decode([FILLER, 1, FILLER, 5, EOS]), "15")
        self.assertEqual(reward((7, 8), [1, 5]), 0.0)          # never ended
        self.assertEqual(reward((7, 8), [1, 6, EOS]), 0.0)


class TestGroupAdvantages(unittest.TestCase):
    def test_rows_sum_to_zero(self):
        r = torch.tensor([[1., 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
        for std in (True, False):
            adv = group_advantages(r, std_normalize=std)
            self.assertTrue(torch.allclose(adv.sum(1), torch.zeros(3), atol=1e-5))
            self.assertTrue(torch.all(adv[2] == 0))   # all correct: no signal, as DAPO notes

    def test_std_division_upweights_easy_and_hard_questions(self):
        r = torch.tensor([[1., 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]])
        grpo = group_advantages(r, std_normalize=True)
        dr = group_advantages(r, std_normalize=False)
        # A rare success gets a larger advantage than a common one under GRPO (difficulty bias) ...
        self.assertGreater(grpo[0, 0], grpo[1, 0])
        self.assertAlmostEqual(grpo[0, 0].item(), (7 / 8) / r[0].std().item(), places=4)
        # ... and every group has the same spread, while Dr. GRPO keeps the raw r - mean.
        self.assertAlmostEqual(grpo[0].std().item(), grpo[1].std().item(), places=4)
        self.assertAlmostEqual(dr[0, 0].item(), 7 / 8, places=6)


class TestLengthNormalization(unittest.TestCase):
    def test_grpo_weights_short_responses_more(self):
        mask = torch.zeros(2, MAX_LEN)
        mask[0, :2] = 1    # a 2-token response
        mask[1, :8] = 1    # an 8-token response
        w = token_weights(mask, length_normalize=True)
        self.assertAlmostEqual(w[0, 0].item(), 1 / 2 / 2)
        self.assertAlmostEqual(w[1, 0].item(), 1 / 8 / 2)
        # Each response weighs the same in total, so a long wrong answer is penalized less per token.
        self.assertAlmostEqual(w[0].sum().item(), w[1].sum().item())
        wd = token_weights(mask, length_normalize=False)
        self.assertAlmostEqual(wd[0, 0].item(), wd[1, 0].item())

    def test_clipped_loss_has_no_gradient_past_the_clip(self):
        lp_old = torch.zeros(1, 3)
        lp_new = torch.full((1, 3), 0.5, requires_grad=True)   # ratio e^0.5 > 1.2
        policy_loss(lp_new, lp_old, torch.tensor([1.0]), torch.ones(1, 3)).backward()
        self.assertTrue(torch.all(lp_new.grad == 0))


class TestTraining(unittest.TestCase):
    def test_grpo_raises_reward_above_the_start(self):
        torch.manual_seed(0)
        pol = Policy()
        sft(pol, split(0)[0], seed=0)
        start, _, _ = evaluate(pol, prompts())
        _, mask = pol.logprobs(torch.tensor([1]), torch.tensor([2]), torch.tensor([target((1, 2)) + [EOS] * 6]))
        self.assertEqual(mask.sum().item(), 2)   # '3' and the end token
        train(pol, prompts(), steps=150, seed=0)
        end, _, _ = evaluate(pol, prompts())
        self.assertGreater(end, start + 0.1)


if __name__ == "__main__":
    unittest.main()
