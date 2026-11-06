"""A small synthetic web crawl with the problems real crawls have (deck 05).

Each document is labeled with what it is, so every filter in this lab can be scored:
  "prose"      sentences built from a small grammar: the text we want
  "boilerplate" navigation menus, cookie banners, keyword lists: short lines, no punctuation
  "repetitive" one line repeated many times
  "near_dup"   a prose document with a few words changed (mirrors, reposts)
  "exact_dup"  a byte-identical copy of an earlier prose document
  "spam"       full sentences stuffed with shopping keywords: passes the line rules, fails a classifier
"""
from __future__ import annotations

import random

SUBJECTS = ["The river", "A small engine", "The committee", "Each student", "The old library", "A new vaccine",
            "The city council", "Our telescope", "The orchestra", "A careful reader", "The bridge", "The algorithm"]
VERBS = ["measures", "changes", "explains", "protects", "connects", "records", "describes", "improves",
         "supports", "reveals", "follows", "reduces"]
OBJECTS = ["the flow of water", "the cost of energy", "how cells divide", "the history of the town",
           "the speed of light", "a pattern in the data", "the rules of the game", "the temperature at night",
           "the shape of the coast", "an error in the proof", "the growth of trees", "the price of bread"]
TAILS = ["after the storm.", "in every season.", "with great care.", "for the first time.", "over many years.",
         "during the experiment.", "without any help.", "across the region.", "before the meeting.", "at a low cost."]
MENU = ["Home", "About us", "Contact", "Login", "Sign up", "Privacy", "Terms", "Shop", "Cart", "FAQ", "Blog", "Careers"]
KEYWORDS = ["cheap", "best", "free", "deal", "online", "buy", "discount", "sale", "top", "offer", "2026", "now"]


def prose(rng: random.Random, n_sentences: int) -> str:
    lines = []
    for _ in range(rng.randint(3, 5)):
        sents = [f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}"
                 for _ in range(max(1, n_sentences // 4))]
        lines.append(" ".join(sents))
    return "\n".join(lines)


def boilerplate(rng: random.Random) -> str:
    lines = [rng.choice(MENU) for _ in range(rng.randint(8, 15))]
    lines += [" ".join(rng.choice(KEYWORDS) for _ in range(rng.randint(3, 6))) for _ in range(rng.randint(2, 5))]
    return "\n".join(lines)


def repetitive(rng: random.Random) -> str:
    line = f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}"
    return "\n".join([line] * rng.randint(8, 20))


def spam(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(3, 5)):
        kw = " ".join(rng.choice(KEYWORDS) for _ in range(rng.randint(4, 8)))
        lines.append(f"{rng.choice(SUBJECTS)} {kw} {rng.choice(KEYWORDS)} {rng.choice(TAILS)} Click {kw} now.")
    return "\n".join(lines)


def near_duplicate(rng: random.Random, doc: str, n_edits: int = 2) -> str:
    words = doc.split(" ")
    for _ in range(n_edits):
        words[rng.randrange(len(words))] = rng.choice(["indeed", "also", "quietly", "often"])
    return " ".join(words)


def make_corpus(n_docs: int = 600, seed: int = 0) -> list[tuple[str, str]]:
    """Returns (label, text) pairs in crawl order."""
    rng = random.Random(seed)
    docs: list[tuple[str, str]] = []
    originals: list[str] = []
    for _ in range(n_docs):
        r = rng.random()
        if r < 0.55 or not originals:
            text = prose(rng, rng.randint(8, 16))
            originals.append(text)
            docs.append(("prose", text))
        elif r < 0.70:
            docs.append(("boilerplate", boilerplate(rng)))
        elif r < 0.78:
            docs.append(("repetitive", repetitive(rng)))
        elif r < 0.88:
            docs.append(("near_dup", near_duplicate(rng, rng.choice(originals))))
        elif r < 0.95:
            docs.append(("spam", spam(rng)))
        else:
            docs.append(("exact_dup", rng.choice(originals)))
    return docs
