"""A toy arithmetic task with a verifiable reward (decks 25 and 27).

Prompts are pairs (a, b) of digits; the answer is a + b written in decimal. A response is a sequence of
tokens from a 12-token vocabulary: the digits, a filler token '.' that stands for "thinking" and carries no
meaning, and an end token. The reward is a program: 1 if the digits before the end token spell a + b,
0 otherwise. No reward model, so nothing to over-optimize except the checker itself.
"""
import random

DIGITS = [str(d) for d in range(10)]
FILLER, EOS = 10, 11
VOCAB = DIGITS + [".", "<eos>"]
MAX_LEN = 8


def prompts():
    return [(a, b) for a in range(10) for b in range(10)]


def split(seed=0, n_sft=30):
    """A few prompts for the SFT warm-up; the RL stage and the evaluation use all 100."""
    ps = prompts()
    random.Random(seed).shuffle(ps)
    return ps[:n_sft], ps[n_sft:]


def decode(tokens):
    """Text of a response up to the end token, fillers dropped."""
    out = []
    for t in tokens:
        if t == EOS:
            break
        if t != FILLER:
            out.append(VOCAB[t])
    return "".join(out)


def reward(prompt, tokens):
    """Rule-based reward: 1.0 if the response ends and its digits equal a + b, else 0.0."""
    a, b = prompt
    if EOS not in tokens:
        return 0.0
    return 1.0 if decode(tokens) == str(a + b) else 0.0


def target(prompt, n_filler=0):
    """A correct response with n_filler '.' tokens first, used only for the SFT warm-up."""
    return [FILLER] * n_filler + [int(c) for c in str(sum(prompt))] + [EOS]
