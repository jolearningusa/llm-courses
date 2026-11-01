"""Lab L2: byte-level BPE from scratch. Standard library only.

    python bpe.py corpus.txt --merges 300

Training follows Sennrich et al. (2015, Algorithm 1) on UTF-8 bytes instead of
characters, as GPT-2 does: the base vocabulary is the 256 byte values, so any
string can be encoded and decoding is lossless. Like GPT-2, text is first split
into chunks (letters, digits, spaces, punctuation) and merges never cross a
chunk boundary.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def pre_split(text: str) -> list[str]:
    """Split text into chunks so merges do not cross character categories (GPT-2 §2.2).

    A chunk is an optional leading space followed by a run of letters (any script),
    a run of digits, or a run of other non-space characters; leftover spaces form
    their own chunk. Joining the chunks gives back the text exactly.
    """
    return re.findall(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+", text)


class ByteBPE:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []          # in the order they were learned
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, num_merges: int) -> "ByteBPE":
        words = Counter(tuple(chunk.encode("utf-8")) for chunk in pre_split(text))
        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in words.items():
                for a, b in zip(word, word[1:]):
                    pairs[a, b] += freq
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))   # most frequent; ties by smallest ids
            new_id = 256 + len(self.merges)
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = Counter({self._merge(w, best, new_id): f for w, f in words.items()})
        return self

    @staticmethod
    def _merge(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        out, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in pre_split(text):
            word = tuple(chunk.encode("utf-8"))
            for k, pair in enumerate(self.merges):         # replay merges in learned order
                if len(word) < 2:
                    break
                word = self._merge(word, pair, 256 + k)
            ids.extend(word)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8")


def chars_per_token(bpe: ByteBPE, text: str) -> float:
    return len(text) / max(1, len(bpe.encode(text)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", nargs="?", default=str(Path(__file__).with_name("corpus")))
    ap.add_argument("--merges", type=int, default=300)
    args = ap.parse_args(argv)
    path = Path(args.corpus)
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    texts = {f.stem: f.read_text(encoding="utf-8") for f in files}
    bpe = ByteBPE().train("\n".join(texts.values()), args.merges)
    print(f"vocabulary: 256 bytes + {len(bpe.merges)} merges")
    print("first merges:", [bpe.vocab[256 + k].decode("utf-8", "replace") for k in range(min(10, len(bpe.merges)))])
    for name, text in texts.items():
        print(f"{name:10} {len(text):6} chars  {chars_per_token(bpe, text):5.2f} chars/token  "
              f"{len(text.encode('utf-8')) / len(text):4.2f} bytes/char  {unicodedata.name(text.strip()[0], '?')[:20]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
