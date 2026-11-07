import unittest
from math import comb

import numpy as np

from passk import pass_at_k, pass_at_k_comb, pass_at_k_brute, pass_at_k_naive
from ratings import simulate_votes, fit_bt, elo
from contamination import make_benchmark, train_corpus, Bigram, ngram_overlap, permutation_test


class TestPassAtK(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for n in (5, 8):
            for c in range(n + 1):
                results = [True] * c + [False] * (n - c)
                rng.shuffle(results)
                for k in range(1, n + 1):
                    b = pass_at_k_brute(results, k)
                    self.assertAlmostEqual(pass_at_k(n, c, k), b, places=12)
                    self.assertAlmostEqual(pass_at_k_comb(n, c, k), b, places=12)

    def test_unbiased_in_expectation(self):
        n = 20
        for p in (0.05, 0.3, 0.7):
            pmf = [comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)]
            for k in (1, 5, 10, 20):
                truth = 1 - (1 - p) ** k
                unb = sum(w * pass_at_k(n, c, k) for c, w in enumerate(pmf))
                naive = sum(w * pass_at_k_naive(n, c, k) for c, w in enumerate(pmf))
                self.assertAlmostEqual(unb, truth, places=12)
                if k > 1:
                    self.assertLess(naive, truth)


class TestRatings(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.true = np.array([1.0, 0.6, 0.1, -0.3, -1.4])
        self.true -= self.true.mean()
        self.w, self.l = simulate_votes(self.true, 20000, rng)
        self.perm = rng.permutation(len(self.w))

    def test_bt_recovers_strengths(self):
        s = fit_bt(self.w, self.l, 5)
        self.assertLess(np.abs(s - self.true).max(), 0.06)

    def test_bt_is_a_maximum(self):
        s = fit_bt(self.w, self.l, 5)
        p = 1 / (1 + np.exp(-(s[self.w] - s[self.l])))
        g = np.zeros(5)
        np.add.at(g, self.w, 1 - p)
        np.add.at(g, self.l, -(1 - p))
        self.assertLess(np.abs(g).max(), 1e-6)

    def test_bt_ignores_order_elo_does_not(self):
        a, b = fit_bt(self.w, self.l, 5), fit_bt(self.w[self.perm], self.l[self.perm], 5)
        self.assertLess(np.abs(a - b).max(), 1e-9)
        e1, e2 = elo(self.w, self.l, 5), elo(self.w[self.perm], self.l[self.perm], 5)
        self.assertGreater(np.abs(e1 - e2).max(), 1e-3)


class TestContamination(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.bench = make_benchmark(100, 12, self.rng)

    def test_clean_model_passes_both_checks(self):
        corpus = train_corpus(100_000, self.bench, 0, self.rng)
        self.assertEqual(ngram_overlap(self.bench, corpus), 0.0)
        self.assertGreater(permutation_test(Bigram(corpus), self.bench, 99, self.rng), 0.05)

    def test_one_ordered_copy_is_detected(self):
        corpus = train_corpus(100_000, self.bench, 1, self.rng)
        self.assertEqual(ngram_overlap(self.bench, corpus), 1.0)
        self.assertEqual(permutation_test(Bigram(corpus), self.bench, 99, self.rng), 0.01)

    def test_shuffled_copy_escapes_the_permutation_test(self):
        shuffled = [self.bench[i] for i in self.rng.permutation(len(self.bench))]
        corpus = train_corpus(100_000, shuffled, 2, self.rng)
        self.assertEqual(ngram_overlap(self.bench, corpus), 1.0)
        self.assertGreater(permutation_test(Bigram(corpus), self.bench, 99, self.rng), 0.05)


if __name__ == "__main__":
    unittest.main()
