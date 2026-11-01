# Lab L9: residual vector quantization of audio

Used with decks 41 and 42. CPU only, NumPy only.

SoundStream and EnCodec (deck 41) compress each encoder frame with a cascade of codebooks: the first
quantizes the frame, the second quantizes what the first missed, and so on. Their codes are also the tokens
that AudioLM and VALL-E generate (deck 42). In this lab the "encoder" is plain framing (each frame is 16
samples of a toy signal), so the residual vector quantizer is the only part that learns.

| Code | Idea | Deck |
|---|---|---|
| `synth_audio()` | A toy voice-like signal: harmonics with a gliding pitch, a loudness envelope and noise | – |
| `frames()`, `unframe()` | Signal ↔ frames of d samples (a stand-in for the convolutional encoder) | 41 |
| `kmeans()`, `nearest()` | Codebook learning and lookup | 41 |
| `RVQ` | One codebook per stage, fitted on the residual of the earlier stages; encode and decode with the first n | 41 |
| `bitrate()` | Frames per second × codebooks × log2(codebook size) | 41, 42 |

## Run

```bash
python -m unittest test_rvq.py   # about 10 s
python rvq.py
```

`python rvq.py` fits 8 codebooks of 64 on 4 s of signal and evaluates on 2 s it has not seen. Our run:

| Codebooks | MSE | SNR (dB) | kbps at 1,000 frames/s |
|---|---|---|---|
| 1 | 6.18e-04 | 18.11 | 6 |
| 2 | 1.69e-04 | 23.73 | 12 |
| 4 | 6.86e-05 | 27.66 | 24 |
| 8 | 1.59e-05 | 34.01 | 48 |

The first codebook does the most work; each one after the second adds about 1.6 dB. The toy frames are raw samples, so
the bitrates are much higher than a real codec's; SoundStream's learned encoder makes each frame cover 320
samples instead of 16.

The tests check the bitrates on the decks (EnCodec and VALL-E at 6 kbps, AudioLM's 12 codebooks at 50 Hz);
that the error falls with every codebook added; that the codes for fewer codebooks are a prefix of the full
codes, which is what lets one model serve several bitrates; that each stage quantizes the residual of the
one before; that two codebooks of 32 beat one codebook of 64 with the same number of stored vectors; and
that 8 codebooks pass 30 dB.

## Tasks

1. Plot SNR against the number of codebooks for codebook sizes 16, 64 and 256. At the same bitrate (for
   example 24 bits per frame), which is better: more small codebooks or fewer large ones?
2. A single codebook with the bits of 4 codebooks of 64 would need 64⁴ ≈ 16.8M entries. Why is that out of
   reach, and what does RVQ give up in exchange?
3. Fit the RVQ on one pitch range (set the glide to 120–150 Hz) and test on another (220–240 Hz). How much
   does the SNR drop? What does that say about training codecs on diverse audio?
4. Write the 2 s test signal back as a WAV file (`scipy.io.wavfile` or the `wave` module) after decoding with
   1, 2 and 8 codebooks, and listen to them.
5. VALL-E (deck 42) predicts codebook 1 autoregressively and codebooks 2 to 8 in parallel. Using your
   results, explain why the first codebook is the one worth generating token by token.
6. Count tokens: 10 s of audio at EnCodec's 75 frames/s with 8 codebooks. Compare with the number of text
   tokens in a 10 s spoken sentence (about 25 words).
