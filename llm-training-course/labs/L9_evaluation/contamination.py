"""Two contamination checks on a toy corpus (deck 32).

1. n-gram overlap: flag a test example if any of its n-grams appears in the training corpus. This needs the
   training data, which is often not published.
2. The permutation test of Oren et al. (2023): a benchmark's examples are exchangeable, so a model that never
   saw the benchmark gives its canonical order no more likelihood than a shuffled order. A model trained on
   the benchmark file, in its published order, prefers that order. This needs only log-probabilities.

The "model" is a bigram language model with add-alpha smoothing, trained on random text from a Zipf
vocabulary plus, optionally, copies of the benchmark in its canonical order. A bigram model sees order
only through the pair (last token of one example, first token of the next), which is enough.
"""
from collections import Counter

import numpy as np

V = 2000


def make_text(n_tokens, rng):
    p = 1.0 / np.arange(1, V + 1)
    return list(rng.choice(V, size=n_tokens, p=p / p.sum()))


def make_benchmark(n_examples, length, rng):
    return [make_text(length, rng) for _ in range(n_examples)]


class Bigram:
    def __init__(self, tokens, alpha=0.1):
        self.pairs = Counter(zip(tokens[:-1], tokens[1:]))
        self.first = Counter(tokens[:-1])
        self.alpha = alpha

    def logprob(self, tokens):
        a = self.alpha
        return float(sum(np.log((self.pairs[(x, y)] + a) / (self.first[x] + a * V))
                         for x, y in zip(tokens[:-1], tokens[1:])))


def train_corpus(n_tokens, bench, dup, rng):
    """Random text with `dup` copies of the benchmark, each in canonical order, inserted at random places."""
    text = make_text(n_tokens, rng)
    flat = [t for ex in bench for t in ex]
    for _ in range(dup):
        i = int(rng.integers(0, len(text)))
        text = text[:i] + flat + text[i:]
    return text


def ngram_overlap(bench, corpus, n=8):
    """Fraction of test examples that share at least one n-gram with the corpus."""
    seen = set(zip(*[corpus[i:] for i in range(n)]))
    hit = [any(g in seen for g in zip(*[ex[i:] for i in range(n)])) for ex in bench]
    return float(np.mean(hit))


def permutation_test(model, bench, n_perm, rng):
    """One-sided p-value: share of shuffled orders at least as likely as the canonical order."""
    canon = model.logprob([t for ex in bench for t in ex])
    count = 0
    for _ in range(n_perm):
        order = rng.permutation(len(bench))
        if model.logprob([t for i in order for t in bench[i]]) >= canon:
            count += 1
    return (count + 1) / (n_perm + 1)


def paraphrase(ex, rng, frac=0.3):
    """Replace a fraction of tokens: a crude stand-in for rewording, which defeats exact matching."""
    ex = list(ex)
    for i in rng.choice(len(ex), size=int(frac * len(ex)), replace=False):
        ex[i] = int(rng.integers(0, V))
    return ex


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    bench = make_benchmark(200, 12, rng)
    print("benchmark: 200 examples of 12 tokens; corpus: 200,000 random tokens plus copies of the benchmark")
    print(f"{'copies in the corpus':>25} {'8-gram':>8} {'permutation p':>14}")
    for dup in (0, 1, 2, 4):
        corpus = train_corpus(200_000, bench, dup, rng)
        model = Bigram(corpus)
        print(f"{dup:25d} {ngram_overlap(bench, corpus):8.2f} {permutation_test(model, bench, 199, rng):14.3f}")
    shuffled = [bench[i] for i in rng.permutation(len(bench))]
    corpus = train_corpus(200_000, shuffled, 4, rng)
    model = Bigram(corpus)
    print(f"{'4, in a shuffled order':>25} {ngram_overlap(bench, corpus):8.2f} {permutation_test(model, bench, 199, rng):14.3f}")
    para = [paraphrase(ex, rng) for ex in bench]
    corpus = train_corpus(200_000, para, 4, rng)
    model = Bigram(corpus)
    print(f"{'4, 30% of tokens reworded':>25} {ngram_overlap(bench, corpus):8.2f} {permutation_test(model, bench, 199, rng):14.3f}")
