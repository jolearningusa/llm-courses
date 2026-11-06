import math
import random
import unittest
from collections import Counter

from corpus import make_corpus, near_duplicate, prose
from dedup import (MinHasher, candidate_probability, dedup_exact, dedup_minhash, estimate, exact_key,
                   jaccard, shingles)
from mixture import effective_data, epochs_needed, sample_mixture
from pipeline import run
from quality import failed_rules, keep_top


class TestDedup(unittest.TestCase):
    def test_normalized_exact_key(self):
        self.assertEqual(exact_key("Hello, World 2024!"), exact_key("hello world 1999"))
        self.assertNotEqual(exact_key("hello world"), exact_key("hello there"))

    def test_minhash_tracks_jaccard(self):
        rng = random.Random(1)
        base = [f"w{i}" for i in range(300)]
        mh = MinHasher(512, seed=3)
        for keep in (0.9, 0.6, 0.3):
            a = set(base)
            b = set(rng.sample(base, int(keep * 300))) | {f"x{i}" for i in range(int((1 - keep) * 300))}
            self.assertAlmostEqual(estimate(mh.signature(a), mh.signature(b)), jaccard(a, b), delta=0.07)

    def test_lsh_s_curve(self):
        # FineWeb's 14 bands of 8: at s = (1/b)^(1/r) the candidate chance is about 1 - 1/e.
        s = (1 / 14) ** (1 / 8)
        self.assertAlmostEqual(candidate_probability(s, 14, 8), 1 - (1 - 1 / 14) ** 14, places=12)
        self.assertLess(candidate_probability(0.5, 14, 8), 0.06)
        self.assertGreater(candidate_probability(0.9, 14, 8), 0.99)

    def test_dedup_keeps_first_copy(self):
        rng = random.Random(0)
        a, b = prose(rng, 12), prose(rng, 12)
        docs = [a, b, a, near_duplicate(rng, b, n_edits=1)]
        self.assertEqual(dedup_exact(docs), [0, 1, 3])
        self.assertEqual(dedup_minhash(docs), [0, 1])
        self.assertGreater(jaccard(shingles(b), shingles(docs[3])), 0.75)


class TestQuality(unittest.TestCase):
    def test_rules_on_hand_examples(self):
        self.assertEqual(failed_rules("A full sentence ends here.\nAnother sentence ends here too."), [])
        self.assertIn("short_lines", failed_rules("Home\nAbout\nContact\nLogin"))
        self.assertIn("punctuation", failed_rules("Home\nAbout\nContact\nLogin"))
        self.assertIn("dup_lines", failed_rules("Same line, again and again.\n" * 5))

    def test_keep_top(self):
        self.assertEqual(keep_top([0.1, 0.9, 0.5, 0.7], 0.5), [1, 3])

    def test_pipeline_funnel(self):
        stages = run(verbose=False)
        names = list(stages)
        first, last = Counter(stages[names[0]]), Counter(stages[names[-1]])
        for bad in ("boilerplate", "repetitive", "exact_dup"):
            self.assertEqual(last[bad], 0)
        self.assertLessEqual(last["spam"], 0.05 * first["spam"])
        self.assertLessEqual(last["near_dup"], 0.25 * first["near_dup"])
        self.assertGreaterEqual(last["prose"], 0.85 * first["prose"])
        sizes = [len(v) for v in stages.values()]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


class TestMixture(unittest.TestCase):
    def test_sampled_proportions(self):
        w = {"web": 0.7, "code": 0.2, "math": 0.1}
        c = Counter(sample_mixture(w, 50000, seed=1))
        for k, v in w.items():
            self.assertAlmostEqual(c[k] / 50000, v, delta=0.01)

    def test_epochs(self):
        e = epochs_needed(1e12, {"web": 0.8, "books": 0.2}, {"web": 2e12, "books": 5e10})
        self.assertAlmostEqual(e["web"], 0.4)
        self.assertAlmostEqual(e["books"], 4.0)

    def test_repetition_value(self):
        u = 1e9
        self.assertEqual(effective_data(u, 1), u)
        self.assertGreater(effective_data(u, 4) / (4 * u), 0.9)         # 4 epochs: almost as good as new data
        cap = u * (1 + 15.39)
        self.assertLess(effective_data(u, 200), cap)                     # never more than U (1 + R*)
        self.assertGreater(effective_data(u, 200) / cap, 0.999)
        r = 15.39                                                           # at R = R*, 1 - 1/e of its value
        self.assertAlmostEqual((effective_data(u, 1 + r) - u) / (u * r), 1 - math.exp(-1), places=9)


if __name__ == "__main__":
    unittest.main()
