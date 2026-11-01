# Lab L3: positions

Used with decks 08, 09 and 10. CPU only, NumPy only.

`positions.py` implements the position methods of Part II side by side: sinusoidal embeddings (deck 01),
RoPE (deck 08), ALiBi slopes and biases (deck 09), and Position Interpolation and YaRN's per-frequency
scaling and temperature (deck 10).

## Run

```bash
python -m unittest test_positions.py
```

## Tasks

1. Prove on paper, then check with `rope()`, that the score between a query at m and a key at n depends only
   on n − m. Which line of `rope()` would break this if you changed it?
2. Print `rope_wavelengths(128)` for base 10,000 and base 500,000 (Llama 3). How many pairs complete less than
   one turn inside an 8K window in each case? Why does a larger base help long contexts?
3. Build `alibi_bias(16, 8)` and plot the softmax of one row for each head. Which heads can still attend
   to the first token from position 15?
4. Use `interpolate()` to extend a 4K model to 32K. What is the smallest gap between two neighboring
   positions now? Explain the short-context regression the PI paper reports.
5. Plot `yarn_ramp(128, 4096)`. Which pairs does YaRN leave untouched, and which does it treat exactly like PI?

## Checks

RoPE scores depend only on the offset (also after interpolation) and rotations preserve norms; the RoPE
frequencies match RoFormer; ALiBi slopes for 8 heads are 1/2 … 1/256 and the bias is linear in distance and
causal; interpolated positions stay inside the trained range; YaRN keeps fast pairs and scales slow pairs
exactly as PI does; and the temperature follows 0.1 ln(s) + 1.
