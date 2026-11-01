import unittest

import numpy as np

from vit_clip import (PatchEmbed, clip_loss, clip_loss_grad, conv2d_strided, interpolate_pos, n_tokens,
                      patchify, train_toy_clip)


class TestPatches(unittest.TestCase):
    def test_token_counts_on_the_decks(self):
        self.assertEqual(n_tokens(224, 224, 16), 197)            # deck 34: 196 patches + [class]
        self.assertEqual(n_tokens(224, 224, 14, cls=False), 256)  # deck 36: LLaVA's 16 x 16 grid

    def test_patchify_keeps_every_pixel_once(self):
        img = np.arange(8 * 8 * 3, dtype=float).reshape(8, 8, 3)
        patches = patchify(img, 4)
        self.assertEqual(patches.shape, (4, 48))
        np.testing.assert_array_equal(np.sort(patches.ravel()), img.ravel())
        np.testing.assert_array_equal(patches[1].reshape(4, 4, 3), img[0:4, 4:8])  # second patch: top right

    def test_patch_embedding_is_a_strided_convolution(self):
        rng = np.random.default_rng(0)
        pe = PatchEmbed(img_size=16, p=4, c=3, d=8, rng=rng)
        img = rng.normal(size=(16, 16, 3))
        linear = patchify(img, 4) @ pe.E + pe.b
        conv = conv2d_strided(img, pe.E.reshape(4, 4, 3, 8), pe.b, stride=4)
        np.testing.assert_allclose(linear, conv, atol=1e-12)
        self.assertEqual(pe(img).shape, (17, 8))  # [class] + 16 patches

    def test_position_interpolation(self):
        rng = np.random.default_rng(1)
        pos = rng.normal(size=(1 + 14 * 14, 8))
        np.testing.assert_allclose(interpolate_pos(pos, 14, 14), pos)        # same grid: unchanged
        big = interpolate_pos(pos, 14, 24)
        self.assertEqual(big.shape, (1 + 24 * 24, 8))
        np.testing.assert_allclose(big[0], pos[0])                             # [class] kept
        g_old, g_new = pos[1:].reshape(14, 14, 8), big[1:].reshape(24, 24, 8)
        for (a, b), (c, d) in [((0, 0), (0, 0)), ((0, 13), (0, 23)), ((13, 13), (23, 23))]:
            np.testing.assert_allclose(g_new[c, d], g_old[a, b])              # corners map to corners


class TestClipLoss(unittest.TestCase):
    def test_identical_embeddings_give_ln_n(self):
        e = np.ones((32, 4))
        loss, _ = clip_loss(e, e)
        self.assertAlmostEqual(loss, np.log(32), places=10)

    def test_perfect_pairing_gives_near_zero(self):
        e = np.eye(8)
        loss, _ = clip_loss(e, 3 * e, scale=100.0)   # CLIP clips the scale at 100
        self.assertLess(loss, 1e-40)                 # ln(1 + 7 e^-100)

    def test_symmetric_in_images_and_captions(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(16, 6)), rng.normal(size=(16, 6))
        self.assertAlmostEqual(clip_loss(a, b)[0], clip_loss(b, a)[0], places=12)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
        ga, gb = clip_loss_grad(a, b, scale=5.0)
        eps = 1e-6
        for x, g in [(a, ga), (b, gb)]:
            for idx in [(0, 0), (3, 2), (5, 4)]:
                old = x[idx]
                x[idx] = old + eps
                up = clip_loss(a, b, 5.0)[0]
                x[idx] = old - eps
                down = clip_loss(a, b, 5.0)[0]
                x[idx] = old
                self.assertAlmostEqual(g[idx], (up - down) / (2 * eps), places=6)

    def test_toy_training_learns_the_pairing(self):
        r = train_toy_clip()
        self.assertLess(r["test_loss"], 0.1 * r["chance_loss"])
        self.assertGreaterEqual(r["test_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
