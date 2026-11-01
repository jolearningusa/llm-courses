import unittest

import numpy as np

from rvq import RVQ, bitrate, frames, kmeans, nearest, run, synth_audio, unframe


class TestRVQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.train = frames(synth_audio(2.0, rng), 16)
        cls.test = frames(synth_audio(1.0, rng), 16)
        cls.rvq = RVQ(n_q=4, k=32).fit(cls.train, rng)

    def test_bitrates_on_the_decks(self):
        self.assertEqual(bitrate(75, 8, 1024), 6000)    # EnCodec / VALL-E: 6 kbps (decks 41-42)
        self.assertEqual(bitrate(75, 2, 1024), 1500)    # EnCodec's lowest setting
        self.assertEqual(bitrate(50, 12, 1024), 6000)   # AudioLM's SoundStream tokens

    def test_framing_round_trip(self):
        x = np.arange(64.0)
        np.testing.assert_array_equal(unframe(frames(x, 16)), x)

    def test_error_falls_as_codebooks_are_added(self):
        errs = [((self.test - self.rvq.decode(self.rvq.encode(self.test, n))) ** 2).mean() for n in range(1, 5)]
        self.assertTrue(all(b < a for a, b in zip(errs, errs[1:])), errs)

    def test_codes_are_a_prefix(self):
        """Dropping the last codebooks gives the lower bitrate (quantizer dropout, deck 41)."""
        full = self.rvq.encode(self.test)
        np.testing.assert_array_equal(self.rvq.encode(self.test, 2), full[:, :2])
        self.assertTrue(((full >= 0) & (full < 32)).all())

    def test_each_stage_quantizes_the_residual(self):
        codes = self.rvq.encode(self.test)
        first = self.rvq.codebooks[0][codes[:, 0]]
        r1 = self.test - first
        np.testing.assert_array_equal(codes[:, 1], nearest(r1, self.rvq.codebooks[1]))

    def test_residual_codebooks_beat_one_codebook_of_the_same_size(self):
        """2 codebooks of 32 (64 stored vectors, 10 bits) against 1 codebook of 64 vectors (6 bits)."""
        one = kmeans(self.train, 64, np.random.default_rng(1))
        err_one = ((self.test - one[nearest(self.test, one)]) ** 2).mean()
        err_two = ((self.test - self.rvq.decode(self.rvq.encode(self.test, 2))) ** 2).mean()
        self.assertLess(err_two, err_one)

    def test_demo_reaches_30_db(self):
        rows = run(n_q=8)
        self.assertGreater(rows[-1][2], 30.0)
        self.assertGreater(rows[-1][2] - rows[0][2], 10.0)


if __name__ == "__main__":
    unittest.main()
