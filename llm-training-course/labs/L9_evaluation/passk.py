"""pass@k from n samples per problem (deck 32).

Chen et al. (2021) sample n >= k programs per problem, count the c that pass the unit tests, and estimate
the chance that at least one of k samples passes as

    pass@k = 1 - C(n - c, k) / C(n, k)

which is unbiased: it is exactly the fraction of the C(n, k) subsets of size k that contain a passing
sample. Plugging p = c / n into 1 - (1 - p)^k instead underestimates pass@k on average.
"""
from itertools import combinations
from math import comb

import numpy as np


def pass_at_k(n, c, k):
    """Unbiased estimator, in the numerically stable product form of Chen et al.'s Figure 3."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_comb(n, c, k):
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_brute(results, k):
    """Fraction of all size-k subsets of the n results (booleans) that contain at least one pass."""
    subsets = list(combinations(range(len(results)), k))
    return sum(any(results[i] for i in s) for s in subsets) / len(subsets)


def pass_at_k_naive(n, c, k):
    """The plug-in estimate 1 - (1 - c/n)^k: consistent, but biased low for a finite n."""
    return 1.0 - (1.0 - c / n) ** k


def bias(p, n, k, trials, rng):
    """Average error of both estimators against the true 1 - (1 - p)^k, over simulated problems."""
    truth = 1.0 - (1.0 - p) ** k
    cs = rng.binomial(n, p, size=trials)
    unb = np.mean([pass_at_k(n, c, k) for c in cs]) - truth
    nai = np.mean([pass_at_k_naive(n, c, k) for c in cs]) - truth
    return truth, unb, nai


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20
    print(f"n = {n} samples per problem, 20,000 simulated problems per row")
    print(f"{'p':>5} {'k':>4} {'true':>7} {'unbiased err':>13} {'naive err':>10}")
    for p in (0.05, 0.2, 0.5):
        for k in (1, 5, 10):
            t, u, a = bias(p, n, k, 20000, rng)
            print(f"{p:5.2f} {k:4d} {t:7.3f} {u:13.4f} {a:10.4f}")
