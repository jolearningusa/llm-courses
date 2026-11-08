import copy
import itertools
import unittest

import torch

from coverage_ablation import fixed_split, measure, pass_at_k, state_hash, strict_check, width_reward
from grpo import train
from policy import Policy
from task import EOS, FILLER, reward, target


class TestCoverageControls(unittest.TestCase):
    def test_split_and_nested_coverage(self):
        train_ps, test_ps = fixed_split()
        self.assertEqual((len(train_ps), len(test_ps)), (80, 20))
        self.assertFalse(set(train_ps) & set(test_ps))
        self.assertEqual(len(set(train_ps + test_ps)), 100)
        self.assertEqual(fixed_split(), (train_ps, test_ps))
        self.assertTrue(set(train_ps[:10]) < set(train_ps[:40]))

    def test_shaping_attack_is_not_task_success(self):
        self.assertEqual(width_reward((1, 2), [9, EOS]), 0.5)
        self.assertEqual(strict_check((1, 2), [9, EOS]), 0)
        self.assertEqual(width_reward((7, 8), [1, 6, EOS]), 0.5)
        self.assertEqual(strict_check((7, 8), [1, 6, EOS]), 0)
        self.assertEqual(width_reward((1, 2), [3]), 0)
        self.assertEqual(width_reward((1, 2), [EOS]), 0)
        self.assertEqual(strict_check((1, 2), [0, 3, EOS]), 0)
        self.assertEqual(strict_check((1, 2), [3, EOS, 9]), 1)

    def test_independent_oracle_agrees_on_task_grammar(self):
        for a in range(10):
            for b in range(10):
                self.assertEqual(strict_check((a, b), target((a, b), 2)), 1)
                self.assertEqual(width_reward((a, b), target((a, b))), 1)
        for tokens in itertools.product([0, 3, 9, FILLER, EOS], repeat=3):
            self.assertEqual(strict_check((1, 2), tokens), reward((1, 2), tokens))

    def test_pass_at_k_against_enumerated_subsets(self):
        values = [True, False, False, True, False]
        for k in range(1, 6):
            subsets = list(itertools.combinations(values, k))
            expected = sum(any(s) for s in subsets) / len(subsets)
            self.assertAlmostEqual(pass_at_k(5, 2, k), expected)
        self.assertEqual(pass_at_k(16, 0, 8), 0)
        self.assertEqual(pass_at_k(16, 16, 8), 1)
        with self.assertRaises(ValueError):
            pass_at_k(4, 2, 8)

    def test_no_signal_trace_and_sampling_cost(self):
        torch.manual_seed(4)
        policy = Policy(hidden=8)
        before = state_hash(policy)
        trace = []
        train(policy, [(1, 2)], steps=2, prompts_per_step=2, G=4,
              reward_fn=lambda p, t: 0.5, trace=trace)
        self.assertEqual(state_hash(policy), before)  # fresh Adam; no prior momentum
        self.assertEqual(len(trace), 2)
        for row in trace:
            self.assertEqual(row['zero_variance_groups'], 2)
            self.assertEqual(row['positive_reward_events'], 8)
            self.assertEqual(row['sampled_token_slots'], 8 * 8)
            self.assertLessEqual(row['valid_tokens'], row['sampled_token_slots'])

    def test_trace_does_not_change_default_training(self):
        torch.manual_seed(9)
        first = Policy(hidden=8)
        second = copy.deepcopy(first)
        train(first, [(1, 2), (7, 8)], steps=2, seed=3)
        trace = []
        train(second, [(1, 2), (7, 8)], steps=2, seed=3, reward_fn=reward, trace=trace)
        self.assertEqual(state_hash(first), state_hash(second))
        metrics = measure(first, [(1, 2)], seed=6)
        self.assertEqual(metrics, measure(first, [(1, 2)], seed=6))
        self.assertLessEqual(metrics['pass_at_k']['1'], metrics['pass_at_k']['8'])


if __name__ == '__main__':
    unittest.main()
