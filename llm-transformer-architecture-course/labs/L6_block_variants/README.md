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


## Optional Mamba-3 algebra supplement

Used with deck 23 slides 9–10. This is separate from the decoder-only model and historical training table above. `ssm_recurrence.py` uses Python 3.9+ and the standard library; it neither imports PyTorch nor installs Mamba.

```bash
python -m unittest test_ssm_recurrence.py -v
python ssm_recurrence.py
```

### Contract and independent reference

For a projected input `v=B*x`, study `h[t]=alpha[t]*h[t-1]+beta[t]*v[t-1]+gamma[t]*v[t]`. The real fixture uses `alpha=exp(delta*A)`, `beta=(1-lambda)*delta*alpha`, `gamma=lambda*delta`. Its complex variant multiplies alpha by `exp(i*phase)`; `phase` is already a discrete per-step angle. See the paper's equations (5)–(6); the toy has one complex coordinate, equivalent to two real coordinates, rather than a learned full layer.

The new local implementation has two independent evaluation paths: `stream()` feeds state forward; `reference()` recomputes each output from the initial state and weighted prior inputs with explicit products of alpha (O(T²)). It is not the optimized SSD/chunked training algorithm. Coefficients are supplied fixture inputs, not learned projections. Initial state and previous input default to zero and can be set explicitly.

The chunk carry is **both `(state, previous projected input)`**. Forgetting the previous input loses the boundary contribution when beta is nonzero. The tests preserve it across 257 inputs in chunks of 17, then deliberately omit it to expose the error. With lambda=1, beta=0, so that previous-input contribution disappears. No arbitrary-lambda claim of second-order ODE accuracy is made: the paper's error-order condition is narrower than its learned parameterization.

For the literal real fixture delta=1, A=0, lambda=0.5, inputs [2,4] and zero carry, the states must be [1,4]. For a fixed 128-step random complex fixture (seed 17), streamed and independently unrolled values differed by at most **6.69e-16** in the executed Python 3.9.6 run. Tests use explicit `abs(error) <= 1e-11 + 1e-11*abs(reference)`; that double-precision fixture tolerance is not a mixed-precision kernel or learned-quality criterion.

### Separate MIMO shape exercise

`mimo_update()` and `mimo_read()` isolate the one-step real algebra `H'=a*H+B*X^T`, `Y=H'^T*C`. They omit the trapezoidal previous-drive term, phase rotations, learned projections, biases/norms, residual/gate and output projection. Do not chain these helpers and call the result a reproduced Mamba-3 block.

| Quantity | Shape in the isolated fixture |
|---|---|
| Core recurrent state H | N × P |
| Current projected B, X, C | N × R, P × R, N × R |
| Intermediate read output Y | P × R |

One shared state receives R outer-product terms; R is a per-token rank, not R future tokens. Increasing rank does not replicate this core state, but it increases input/read work and other storage. For N=P=2, the literal rank-two test gives updated H `[[17.5,24],[40.5,55]]`; identity C reads its transpose. Rank-one and rank-two states both contain four scalars. This is shape/algebra evidence only.

### What to measure next

- Read the [pinned author module](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py). Trace `step`, including the phase/SSM/K/V caches, and list every allocated tensor and dtype. Its K cache depends on rank; core-state size alone is not total cache bytes. The step comment describes H100 testing, not universal device support.
- Before comparing kernels, freeze batch, sequence lengths, rank, state/head dimensions, precision, shapes and numerical tolerance. Record supported hardware/source/build identity and separate prefill/decode. Count state traffic, projections, rotations, auxiliary caches and workspace. Match quality/parameter budgets before interpreting model comparisons.
- State tracking and retrieval need trained-checkpoint tasks. The rotation identity and random recurrence agreement do not demonstrate learned parity, recall or perplexity. No paper gain, GPU latency, arithmetic-intensity measurement or production speedup was reproduced here.

Validation: eight new tests passed on Python 3.9.6. All six existing L6 model tests plus the eight new tests also passed on Python 3.12.14 / torch 2.9.1 CPU (one thread); PyTorch warned NumPy was absent, but no NumPy call was used. This is not a re-run of the historical 400-step/three-seed training tables.

### Original paper-critique questions

The revised closing pages retain these questions here: why does time invariance fail selective copying? What do very large/small delta do? Why does selection require a scan, and how is it parallelized? What does Mamba-2 trade away by making A scalar? Give a bounded-state recall task and state the finite-precision/state-capacity assumptions behind an estimated failure length. Original Mamba/Mamba-2 evidence remains on deck 23 slides 6–8; hybrids and Gated DeltaNet remain decks 24–25.
