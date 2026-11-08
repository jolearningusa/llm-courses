# Lab L5: attention variants and the KV cache

Used with decks 12 to 16. CPU only, NumPy only.

`attention.py` holds one attention module per idea in Part III:

| Code | Idea | Deck |
|---|---|---|
| `GroupedAttention(n_kv_heads=...)` | MHA (`n_kv_heads = n_heads`), GQA, MQA (`n_kv_heads = 1`), with a KV cache | 12, 13 |
| `kv_bytes_per_token()` | 2 × KV heads × head dim × layers × bytes | 12, 13 |
| `mean_pool_to_gqa()` | Convert an MHA module to GQA by averaging each group's key and value projections | 13 |
| `blockwise_attention()` | Exact attention one block of keys at a time, with a running max and sum | 14 |
| `window_mask(window=, sinks=)` | Causal, sliding-window and sink-plus-window masks | 15 |
| `MLA` | A latent KV cache with decoupled RoPE; decoding with the up-projections absorbed | 16 |

Weights are random: the lab checks that each variant computes what its paper says, not that it has learned anything.

## Run

```bash
python -m unittest test_attention.py
```

The tests check that cached one-token-at-a-time decoding equals full recomputation for MHA, GQA, MQA
and MLA; that the cache sizes and byte counts match the decks; that block-by-block attention equals the
direct computation; and that MLA's absorbed decoding equals its explicit form.

## Tasks

1. In `GroupedAttention.step`, print the cache shapes after 12 tokens for `n_kv_heads` = 8, 2 and 1.
   Relate the ratio to the factor h / G on deck 13.
2. Use `kv_bytes_per_token` with the Lab L0 configs to fill in the table on deck 13 for a 32K-token
   window. Which model fits a batch of 8 such sequences in 24 GB of cache?
3. `test_mean_pooled_model_is_closer_than_random_kv` compares mean pooling with random key and value
   heads. Add the third option from the GQA paper (keep the first head of each group) and rank all three.
4. In `blockwise_attention`, remove the rescaling by `scale`. Which test fails, and why does the running
   maximum make the rescaling necessary?
5. Use `window_mask(n, window=4, sinks=1)` in `GroupedAttention.forward`. How many cached keys does each
   query need? Write a `step` that keeps a rolling buffer of that size (deck 15).
6. In `MLA.step`, the query is multiplied by `W_uk` before it meets the cache. Write the explicit
   version that builds per-head keys from the cached latents, and count the multiplications per step for
   both. Then move RoPE onto the content key and explain why the absorbed form stops working (deck 16, slide 5).

## Context quality is a separate experiment

Use the [L3 controlled-context extension](../L3_positions/#context-evaluation) with deck 15's new closing page. Random-weight cache/recomputation equivalence checks the declared execution relationship; it cannot establish that a trained model retrieves or reasons over old evidence. Log the actual mask, visible information, tokenizer/template lengths and per-cell task outcomes before attributing a quality change to cache correctness. The new fixture builder performs zero model runs.
