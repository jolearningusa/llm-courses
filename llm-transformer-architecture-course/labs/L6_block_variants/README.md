# Lab L6: block variants on a small language model

Used with decks 12, 13, 17 and 18. CPU only, PyTorch 2 (`pip install torch`).

`model.py` is one small decoder-only model whose block is switched by four settings:

| Setting | Values | Deck |
|---|---|---|
| `placement` | `"post"`: LN(x + F(x)), as in 2017; `"pre"`: x + F(LN(x)) with a final norm | 17 |
| `norm` | `"layer"` (LayerNorm) or `"rms"` (RMSNorm) | 17 |
| `ffn` | `"relu"` with 4d hidden, or `"swiglu"` with the parameter-matched hidden size 8d/3 | 18 |
| `n_kv_heads` | 4 (MHA), 2 (GQA), 1 (MQA) | 12, 13 |

`train.py` trains each variant on the same generated data (lines such as `417+86=503;`, 13 characters),
with the same budget and seeds, a constant learning rate and no warm-up, and reports the final
validation loss as mean ± standard deviation over seeds. Loss curves go to `curves.csv`.

## Run

```bash
python -m unittest test_model.py                      # about a minute
python train.py                                       # 6 variants x 3 seeds, 4 layers, lr 3e-3
python train.py --layers 8 --lr 1e-2 --variants post-ln pre-ln pre-rms
```

## Reference results

Our run on a 4-core laptop-class CPU (PyTorch 2.15, 400 steps). Your numbers will differ slightly with
the PyTorch version and the hardware; the pattern should not.

| 4 layers, lr 3e-3 | Parameters | Final validation loss |
|---|---|---|
| post-ln | 203,392 | 1.732 ± 0.006 |
| pre-ln | 203,520 | 1.740 ± 0.012 |
| pre-rms | 202,944 | 1.743 ± 0.011 |
| pre-rms-swiglu | 203,200 | 1.740 ± 0.009 |
| gqa-2 | 186,816 | 1.735 ± 0.009 |
| mqa | 178,624 | 1.744 ± 0.018 |

| 8 layers, lr 1e-2, no warm-up | Parameters | Final validation loss |
|---|---|---|
| post-ln | 401,024 | 2.554 ± 0.000 |
| pre-ln | 401,152 | 1.773 ± 0.017 |
| pre-rms | 400,064 | 1.783 ± 0.021 |

Read them like a paper's table:

- At 4 layers and a moderate learning rate, every variant lands within about one standard deviation of
  the others. At this size and budget the block choice does not show, which is why the papers in
  decks 17 and 18 report multiple runs and train far longer.
- At 8 layers and a large learning rate with no warm-up, Post-LN does not train at all: 2.554 is the
  loss of a uniform guess over 13 characters (ln 13 = 2.565). Pre-LN and Pre-RMSNorm train normally.
  This is the failure Xiong et al. explain (deck 17).
- GQA and MQA save 8% and 12% of the parameters here, all from the key and value projections, at no
  measurable cost; their real saving is the KV cache at inference (lab L5).

## Tasks

1. Plot `curves.csv` (val loss against step, one line per seed). Which variants' curves cross?
2. Repeat the 8-layer run with `--lr 3e-3`. Does Post-LN train? Add a linear warm-up over the first 100
   steps to `train()` and rerun Post-LN at `--lr 1e-2`.
3. Print the hidden-state norm after each block of a Pre-LN and a Post-LN model at initialization. Which
   one grows with depth (deck 17, Lemma 2)?
4. Replace SwiGLU with GEGLU (`F.gelu` in place of `F.silu`). Is the difference larger than the spread
   across seeds?
5. Double `--steps` and the number of seeds. Does any 4-layer difference become larger than its spread?
   State what you would need to claim one variant is better.
