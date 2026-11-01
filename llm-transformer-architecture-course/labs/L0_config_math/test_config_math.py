"""Checks for Lab L0. Run: python -m unittest test_config_math.py"""
import unittest
from pathlib import Path

from config_math import kv_bytes_per_token, load, params, train_flops

CONFIGS = Path(__file__).parent / "configs"

# Parameter counts of the released checkpoints (sum of tensor sizes in their weight files).
EXPECTED_TOTAL = {
    "gpt2": 124_439_808,
    "llama-2-7b": 6_738_415_616,
    "llama-3-8b": 8_030_261_248,
    "mistral-7b-v0.1": 7_241_732_096,
    "qwen2.5-7b": 7_615_616_512,   # model card: 7.61B total, 6.53B non-embedding
}


class ConfigMath(unittest.TestCase):
    def test_total_parameters_match_released_checkpoints(self):
        for name, expected in EXPECTED_TOTAL.items():
            with self.subTest(name):
                self.assertEqual(params(load(CONFIGS / f"{name}.json"))["total"], expected)

    def test_qwen_non_embedding_matches_model_card(self):
        p = params(load(CONFIGS / "qwen2.5-7b.json"))
        self.assertAlmostEqual(p["non_embedding"] / 1e9, 6.53, places=2)

    def test_kv_cache_formula(self):
        # Llama 3 8B: 2 (K and V) x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB per token.
        self.assertEqual(kv_bytes_per_token(load(CONFIGS / "llama-3-8b.json")), 131_072)
        # Llama 2 7B has no GQA: four times as much.
        self.assertEqual(kv_bytes_per_token(load(CONFIGS / "llama-2-7b.json")), 4 * 131_072)

    def test_gpt3_training_compute(self):
        # Deck 05: 175B parameters x 300B tokens x 6 = 3.15e23 FLOPs.
        self.assertAlmostEqual(train_flops(175e9, 300e9) / 1e23, 3.15, places=2)


if __name__ == "__main__":
    unittest.main()
