"""Lab L9: residual vector quantization of an audio signal. NumPy only.

The codecs of deck 41 (SoundStream, EnCodec) quantize each encoder frame with a cascade of codebooks:
the first codebook quantizes the frame, the second quantizes what the first missed, and so on. Here the
"encoder" is plain framing (each frame is a short block of samples) so that the quantizer is the only
thing being learned.

- synth_audio(): a toy signal: a few harmonics with a gliding pitch, an amplitude envelope and noise.
- frames(), unframe(): split a signal into frames and put it back.
- kmeans(): codebook learning.
- RVQ: fit one codebook per stage on the residual of the stages before; encode and decode with the first n.
- bitrate(): frames per second x codebooks x log2(codebook size).

    python rvq.py   # prints reconstruction error and SNR as codebooks are added
"""
from __future__ import annotations

import numpy as np

SR = 16_000


def synth_audio(seconds: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 180 + 60 * np.sin(2 * np.pi * 0.5 * t)               # pitch glides between 120 and 240 Hz
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum(a * np.sin(k * phase) for k, a in [(1, 1.0), (2, 0.5), (3, 0.3), (5, 0.15)])
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t) ** 2          # loudness changes
    return 0.3 * env * x + 0.01 * rng.normal(size=t.size)


def frames(x: np.ndarray, d: int) -> np.ndarray:
    n = len(x) // d
    return x[:n * d].reshape(n, d)


def unframe(f: np.ndarray) -> np.ndarray:
    return f.reshape(-1)


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    """Plain Lloyd's algorithm with k-means++ style seeding. Returns (k, d) centroids."""
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        assign = nearest(X, C)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                C[j] = members.mean(0)
    return C


def nearest(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)
    return d2.argmin(1)


class RVQ:
    def __init__(self, n_q: int, k: int):
        self.n_q, self.k = n_q, k
        self.codebooks: list[np.ndarray] = []

    def fit(self, X: np.ndarray, rng: np.random.Generator) -> "RVQ":
        residual = X.copy()
        self.codebooks = []
        for _ in range(self.n_q):
            C = kmeans(residual, self.k, rng)
            self.codebooks.append(C)
            residual = residual - C[nearest(residual, C)]
        return self

    def encode(self, X: np.ndarray, n: int | None = None) -> np.ndarray:
        """(frames, n) integer codes, one per codebook, using the first n codebooks."""
        n = self.n_q if n is None else n
        residual, codes = X.copy(), []
        for C in self.codebooks[:n]:
            c = nearest(residual, C)
            codes.append(c)
            residual = residual - C[c]
        return np.stack(codes, axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return sum(self.codebooks[i][codes[:, i]] for i in range(codes.shape[1]))


def bitrate(frames_per_second: float, n_codebooks: int, codebook_size: int) -> float:
    return frames_per_second * n_codebooks * np.log2(codebook_size)


def snr_db(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(10 * np.log10((x ** 2).sum() / ((x - x_hat) ** 2).sum()))


def run(d: int = 16, k: int = 64, n_q: int = 8, seed: int = 0):
    """Fit on 4 s of signal, evaluate on 2 s it has not seen. Returns rows (n, mse, snr, kbps)."""
    rng = np.random.default_rng(seed)
    train = frames(synth_audio(4.0, rng), d)
    test_signal = synth_audio(2.0, rng)
    test = frames(test_signal, d)
    rvq = RVQ(n_q, k).fit(train, rng)
    rows = []
    for n in range(1, n_q + 1):
        rec = rvq.decode(rvq.encode(test, n))
        rows.append((n, float(((test - rec) ** 2).mean()), snr_db(test, rec), bitrate(SR / d, n, k) / 1000))
    return rows


def main():
    rows = run()
    print("RVQ on held-out toy audio: frames of 16 samples at 16 kHz (1,000 frames/s), codebooks of 64")
    print(f"{'codebooks':>9} {'MSE':>10} {'SNR dB':>8} {'kbps':>6}")
    for n, mse, snr, kbps in rows:
        print(f"{n:>9} {mse:>10.2e} {snr:>8.2f} {kbps:>6.0f}")
    print(f"\nEnCodec at 24 kHz: 75 frames/s x 8 codebooks x 10 bits = {bitrate(75, 8, 1024) / 1000:.0f} kbps")


if __name__ == "__main__":
    main()
