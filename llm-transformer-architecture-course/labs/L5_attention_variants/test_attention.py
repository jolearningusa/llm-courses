import unittest

import numpy as np

from attention import (GroupedAttention, MLA, attend, blockwise_attention, causal_mask, kv_bytes_per_token,
                       mean_pool_to_gqa, softmax, window_mask)

D, H, N = 32, 8, 12


def decode_all(module, x):
    cache = module.new_cache()
    return np.stack([module.step(x[t], cache) for t in range(len(x))]), cache


class TestGroupedAttention(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(0).normal(size=(N, D))

    def test_cached_decoding_equals_full_recomputation(self):
        for n_kv in (8, 2, 1):                       # MHA, GQA-2, MQA
            with self.subTest(n_kv=n_kv):
                attn = GroupedAttention(D, H, n_kv, np.random.default_rng(1))
                out, cache = decode_all(attn, self.x)
                np.testing.assert_allclose(out, attn.forward(self.x), atol=1e-10)
                self.assertEqual(cache["k"].shape, (n_kv, N, D // H))

    def test_cache_shrinks_by_heads_over_groups(self):
        sizes = {g: GroupedAttention(D, H, g, np.random.default_rng(0)).cache_values_per_token() for g in (8, 2, 1)}
        self.assertEqual(sizes[8] // sizes[2], 4)
        self.assertEqual(sizes[8] // sizes[1], 8)

    def test_kv_bytes_match_deck_numbers(self):
        self.assertEqual(kv_bytes_per_token(32, 128, 32), 512 * 1024)     # Llama 2 7B
        self.assertEqual(kv_bytes_per_token(8, 128, 32), 128 * 1024)      # Llama 3 8B
        self.assertEqual(kv_bytes_per_token(4, 128, 28), 56 * 1024)       # Qwen2.5 7B
        self.assertEqual(kv_bytes_per_token(1, 128, 32), 16 * 1024)       # MQA at Llama 2 7B shape

    def test_mean_pooling_of_identical_heads_changes_nothing(self):
        mha = GroupedAttention(D, H, H, np.random.default_rng(2))
        g = 2
        for W in (mha.Wk, mha.Wv):                   # make heads within each group identical
            blocks = W.reshape(D, g, H // g, D // H)
            blocks[:] = blocks[:, :, :1]
        gqa = mean_pool_to_gqa(mha, g)
        np.testing.assert_allclose(gqa.forward(self.x), mha.forward(self.x), atol=1e-10)

    def test_mean_pooled_model_is_closer_than_random_kv(self):
        rng = np.random.default_rng(3)
        mha = GroupedAttention(D, H, H, rng)
        ref = mha.forward(self.x)
        pooled = mean_pool_to_gqa(mha, 2)
        rand = mean_pool_to_gqa(mha, 2)
        rand.Wk, rand.Wv = rng.normal(0, D ** -0.5, rand.Wk.shape), rng.normal(0, D ** -0.5, rand.Wv.shape)
        err = lambda m: np.abs(m.forward(self.x) - ref).mean()
        self.assertLess(err(pooled), err(rand))


class TestMasksAndBlocks(unittest.TestCase):
    def test_window_and_sinks(self):
        m = window_mask(8, window=3, sinks=1)
        self.assertTrue(m[7, 0])                     # sink kept
        self.assertFalse(m[7, 4])                    # outside the window
        self.assertTrue(m[7, 5] and m[7, 7])
        self.assertFalse(m[2, 3])                    # still causal
        np.testing.assert_array_equal(window_mask(8), causal_mask(8))
        self.assertTrue((window_mask(50, window=4).sum(axis=1) <= 4).all())

    def test_blockwise_equals_direct(self):
        rng = np.random.default_rng(4)
        q, k, v = (rng.normal(size=(N, 16)) for _ in range(3))
        direct = attend(q[None], k[None], v[None], causal_mask(N))[0]
        for block in (1, 3, 5, N):
            np.testing.assert_allclose(blockwise_attention(q, k, v, block), direct, atol=1e-12)
        full = softmax(q @ k.T / 4.0) @ v
        np.testing.assert_allclose(blockwise_attention(q, k, v, 4, causal=False), full, atol=1e-12)


class TestMLA(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(5).normal(size=(N, D))
        self.mla = MLA(D, n_heads=4, head_dim=8, d_c=12, d_rope=4, rng=np.random.default_rng(6))

    def test_absorbed_decoding_equals_explicit_attention(self):
        out, cache = decode_all(self.mla, self.x)
        np.testing.assert_allclose(out, self.mla.forward(self.x), atol=1e-10)
        self.assertEqual(cache["c"].shape, (N, 12))
        self.assertEqual(cache["kr"].shape, (N, 4))

    def test_cache_is_smaller_than_mha_with_same_heads(self):
        self.assertEqual(self.mla.cache_values_per_token(), 16)
        mha = GroupedAttention(D, 4, 4, np.random.default_rng(0))   # same 4 heads of 8
        self.assertEqual(mha.cache_values_per_token(), 64)

    def test_deepseek_v2_numbers(self):
        mla_bytes = (512 + 64) * 60 * 2
        mha_bytes = kv_bytes_per_token(128, 128, 60)
        self.assertEqual(mla_bytes, 69120)           # 67.5 KiB
        self.assertEqual(mha_bytes, 3932160)         # 3.75 MiB
        self.assertAlmostEqual((512 + 64) / (2 * 128), 2.25)   # same as GQA with 2.25 groups


if __name__ == "__main__":
    unittest.main()
