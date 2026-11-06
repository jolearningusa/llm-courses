"""A character-level chat template with a loss mask, and a synthetic instruction dataset.

Deck 15 and architecture deck 11: a conversation is rendered into one token sequence with role
markers, and the SFT loss is taken only where the assistant speaks.

    <|user|>reverse abc<|end|><|assistant|>cba<|end|>
                                           ^^^^^^^^^^  loss on these tokens only
"""
from __future__ import annotations

import random
import string

SPECIALS = ["<pad>", "<|user|>", "<|assistant|>", "<|end|>"]
CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "
VOCAB = SPECIALS + list(CHARS)
ID = {t: i for i, t in enumerate(VOCAB)}
PAD, USER, ASSISTANT, END = (ID[s] for s in SPECIALS)


def encode(text: str) -> list[int]:
    return [ID[c] for c in text]


def decode(ids: list[int]) -> str:
    return "".join(VOCAB[i] for i in ids)


def render(conversation: list[tuple[str, str]]) -> tuple[list[int], list[int]]:
    """Token ids and a mask that is 1 on each assistant reply and the <|end|> that closes it.

    The role marker <|assistant|> itself is not trained: the template always writes it.
    """
    ids, mask = [], []
    for role, text in conversation:
        head = USER if role == "user" else ASSISTANT
        body = encode(text) + [END]
        ids += [head] + body
        mask += [0] + [1 if role == "assistant" else 0] * len(body)
    return ids, mask


def next_token_example(conversation: list[tuple[str, str]]) -> tuple[list[int], list[int], list[int]]:
    """(inputs, targets, loss mask on targets): the mask is shifted with the targets."""
    ids, mask = render(conversation)
    return ids[:-1], ids[1:], mask[1:]


TASKS = {
    "reverse": lambda w: w[::-1],
    "upper": lambda w: w.upper(),
    "sort": lambda w: "".join(sorted(w)),
    "first": lambda w: w[0],
}


def make_conversation(rng: random.Random, max_len: int = 6, turns: int = 1) -> list[tuple[str, str]]:
    conv = []
    for _ in range(turns):
        task = rng.choice(sorted(TASKS))
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, max_len)))
        conv += [("user", f"{task} {word}"), ("assistant", TASKS[task](word))]
    return conv


def dataset(n: int, seed: int, max_turns: int = 2) -> list[list[tuple[str, str]]]:
    rng = random.Random(seed)
    return [make_conversation(rng, turns=rng.randint(1, max_turns)) for _ in range(n)]
