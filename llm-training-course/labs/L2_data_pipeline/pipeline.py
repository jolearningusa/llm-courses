"""Run a crawl through rules, exact dedup, MinHash dedup and a classifier, and print the funnel (decks 05, 06).

    python pipeline.py
    python pipeline.py --docs 2000 --bands 20 --rows 5 --threshold 0.8
"""
from __future__ import annotations

import argparse
from collections import Counter

from corpus import make_corpus
from dedup import dedup_exact, dedup_minhash
from quality import NaiveBayes, failed_rules, keep_top

LABELS = ["prose", "near_dup", "exact_dup", "boilerplate", "repetitive", "spam"]


def row(stage: str, labels: list[str]) -> str:
    c = Counter(labels)
    return f"{stage:<22}{len(labels):>6}" + "".join(f"{c[l]:>12}" for l in LABELS)


def train_classifier(seed: int = 99) -> NaiveBayes:
    """Positives: prose; negatives: everything a crawl adds. Drawn from a separate crawl, like DCLM's
    reference sets, so the classifier never sees the documents it scores."""
    ref = make_corpus(300, seed=seed)
    pos = [t for l, t in ref if l == "prose"]
    neg = [t for l, t in ref if l in ("boilerplate", "repetitive", "spam")]
    return NaiveBayes().fit(pos, neg)


def run(n_docs: int = 600, bands: int = 14, rows: int = 8, threshold: float = 0.75, top: float = 0.9,
        seed: int = 0, verbose: bool = True) -> dict[str, list[str]]:
    docs = make_corpus(n_docs, seed)
    stages = {"crawl": [l for l, _ in docs]}
    docs = [(l, t) for l, t in docs if not failed_rules(t)]
    stages["after rules"] = [l for l, _ in docs]
    docs = [docs[i] for i in dedup_exact([t for _, t in docs])]
    stages["after exact dedup"] = [l for l, _ in docs]
    docs = [docs[i] for i in dedup_minhash([t for _, t in docs], bands, rows, threshold)]
    stages["after MinHash dedup"] = [l for l, _ in docs]
    clf = train_classifier()
    docs = [docs[i] for i in keep_top([clf.score(t) for _, t in docs], top)]
    stages[f"classifier, top {top:.0%}"] = [l for l, _ in docs]
    if verbose:
        print(f"{'stage':<22}{'docs':>6}" + "".join(f"{l:>12}" for l in LABELS))
        for name, labels in stages.items():
            print(row(name, labels))
    return stages


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", type=int, default=600)
    p.add_argument("--bands", type=int, default=14)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--top", type=float, default=0.9)
    a = p.parse_args()
    run(a.docs, a.bands, a.rows, a.threshold, a.top)


if __name__ == "__main__":
    main()
