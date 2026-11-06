"""Quality filtering: rules, then a classifier (decks 05 and 06).

Rules: three of FineWeb's line-level heuristics (arXiv:2406.17557 section 3.6) and a repetition check in
the spirit of Gopher's rules used by RefinedWeb and Dolma. Each returns True when the document should be
dropped.
Classifier: a bag of words and word bigrams with naive Bayes, the same family as the fastText bigram
classifiers of CCNet, DCLM and DeepSeekMath. It is trained on a few positive and negative examples and
scores every document; the pipeline keeps the top fraction, as DCLM keeps its top 10%.
"""
from __future__ import annotations

import math
from collections import Counter


def lines_of(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def few_lines_end_in_punctuation(text: str, max_frac: float = 0.12) -> bool:
    ls = lines_of(text)
    return sum(l[-1] in ".!?\"'" for l in ls) / len(ls) <= max_frac


def many_duplicated_line_chars(text: str, min_frac: float = 0.1) -> bool:
    ls = lines_of(text)
    counts = Counter(ls)
    dup_chars = sum(len(l) for l in ls if counts[l] > 1)
    return dup_chars / sum(len(l) for l in ls) >= min_frac


def many_short_lines(text: str, short: int = 30, min_frac: float = 0.67) -> bool:
    ls = lines_of(text)
    return sum(len(l) < short for l in ls) / len(ls) >= min_frac


RULES = {"punctuation": few_lines_end_in_punctuation, "dup_lines": many_duplicated_line_chars,
         "short_lines": many_short_lines}


def failed_rules(text: str) -> list[str]:
    return [name for name, rule in RULES.items() if rule(text)]


def features(text: str) -> list[str]:
    words = text.lower().split()
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class NaiveBayes:
    """Two-class multinomial naive Bayes with add-one smoothing; score = log P(pos|x) - log P(neg|x)."""

    def fit(self, positives: list[str], negatives: list[str]) -> "NaiveBayes":
        self.counts = {1: Counter(), 0: Counter()}
        for label, docs in ((1, positives), (0, negatives)):
            for d in docs:
                self.counts[label].update(features(d))
        self.totals = {k: sum(c.values()) for k, c in self.counts.items()}
        self.vocab = set(self.counts[1]) | set(self.counts[0])
        self.prior = math.log(len(positives) / len(negatives))
        return self

    def score(self, text: str) -> float:
        v = len(self.vocab) + 1
        s = self.prior
        for f in features(text):
            s += math.log((self.counts[1][f] + 1) / (self.totals[1] + v))
            s -= math.log((self.counts[0][f] + 1) / (self.totals[0] + v))
        return s / max(1, len(features(text)))      # per feature, so long documents are not favored


def keep_top(scores: list[float], fraction: float) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return sorted(order[:max(1, round(fraction * len(scores)))])
