"""Checks for Lab L2. Run: python -m unittest test_bpe.py"""
import unittest
from pathlib import Path

from bpe import ByteBPE, pre_split

CORPUS = {p.stem: p.read_text(encoding="utf-8") for p in (Path(__file__).parent / "corpus").glob("*.txt")}


class ByteLevelBPE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bpe = ByteBPE().train("\n".join(CORPUS.values()), 300)

    def test_pre_split_is_lossless(self):
        for text in CORPUS.values():
            self.assertEqual("".join(pre_split(text)), text)

    def test_round_trip_is_lossless(self):
        samples = list(CORPUS.values()) + ["unseen: Ωmega 🙂 naïve  double  spaces\n\ttab", "数字 2026-11-01"]
        for text in samples:
            self.assertEqual(self.bpe.decode(self.bpe.encode(text)), text)

    def test_never_unknown(self):
        # Every id is a byte or a learned merge: there is no unknown token.
        ids = self.bpe.encode("ẞ Ǆ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢")
        self.assertTrue(all(i in self.bpe.vocab for i in ids))

    def test_merges_compress(self):
        text = CORPUS["en"]
        self.assertLess(len(self.bpe.encode(text)), len(text.encode("utf-8")) / 2)

    def test_frequent_word_becomes_one_token(self):
        bpe = ByteBPE().train("the cat and the dog and the bird " * 20, 20)
        self.assertEqual(len(bpe.encode(" the")), 1)

    def test_language_cost_differs(self):
        # Deck 07: the corpus files say the same thing in four languages. Chinese is a small share of the
        # training text and costs ~3 bytes per character, so the same content needs more tokens.
        en, zh = (len(self.bpe.encode(CORPUS[k])) for k in ("en", "zh"))
        self.assertGreater(zh, en)

if __name__ == "__main__":
    unittest.main()
