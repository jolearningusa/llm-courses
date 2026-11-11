# Shared worksheet: shapes, gradients and decoder causality — 2026-11-10

FND-3 integration from main `989f008` after #136. Use with foundations deck 20 slides 9/10, [foundations L1](../labs/L1_autograd_mlp/), [architecture L1](../../llm-transformer-architecture-course/labs/L1_one_block/), and [training L1](../../llm-training-course/labs/L1_pretraining_loop/). This is a small bridge between existing labs, not a replacement language-model course.

## 1. Predict shapes before running

Fill in the missing output/gradient shapes. For a tokenwise MLP, flatten batch and time to `N=B×T`; keep the feature axis distinct from token axes.

| Step | Input / weights | Contract to explain |
|---|---|---|
| MLP branch | `X:(N,d)`, `W1:(d,f)`, `b1:(f)`, `W2:(f,d)` | Hidden `(N,f)`, branch `(N,d)`; the same weights apply to each position |
| Residual addition | `Y=X+F(X)` | Both terms must align as `(N,d)`; broadcasting can accidentally pass shapes while violating the intended computation |
| Backward path | Upstream `G:(N,d)` | `dL/dX=G+J_F(X)^T G`; with X/G vectorized for the Jacobian notation; include the identity contribution, not only the branch |
| Split self-attention | `X:(T,d)`, `h` heads, `k=d/h` | Q/K/V `(h,T,k)`; scores `(h,T,T)`; merged output `(T,d)` |
| Output head | Hidden `(T,d)`, tied embedding `(V,d)` in architecture L1 | Logits `(T,V)`; training L1 uses `(B,T,V)` and a separate output head |
| Shifted next-token loss | Inputs `tokens[i:i+T]`, targets `tokens[i+1:i+T+1]` | Logit at position `t` predicts the next symbol. Causal masking alone does not verify label shifting or document packing |

In architecture L1's actual default model, `T=8,d=32,h=4,k=8,f=64,V=50`, so logits are `(8,50)`. The small residual derivative fixture separately uses `X:(2,3)`, `W:(3,3)`, `F(X)=tanh(XW)` and `L=sum(Y)`; it is a smooth derivative example, not the complete Transformer backward pass. The zero-weight branch gives `dL/dX=ones` for this loss.

The course block is **pre-LN**: normalize before each branch, then add the residual. The original Transformer paper uses post-LN. Do not silently interchange their forward or derivative equations. Layer normalization over the feature axis is also different from mixing statistics across future token positions.

## 2. Match each check to its oracle

| Existing check / exercise | What passing supports | What it does not establish |
|---|---|---|
| Foundations `Autograd` tests: centered differences, reused values, broadcast summation | These backward computations match their declared numerical or literal references at the tested inputs | Decoder causality, a universal tolerance or correctness of every gradient |
| Smooth residual fixture | Input/weight derivatives and identity path at the recorded point | LayerNorm/attention/whole-Transformer backward validation |
| Architecture `test_attention_matches_definition` | Vectorized attention matches the loop with the same specified mask | Whether the decoder call site supplies the correct mask |
| Architecture `test_decoder_never_sees_the_future` | Changing a suffix preserves prefix logits for this random-weight input | Long-context understanding or all possible inputs/configurations |
| Training `TestModel.test_causal` | Prefix invariance for the tested PyTorch decoder configuration | Correct next-token target shift, packing boundaries or model quality |
| Training `TestModel.test_z_loss` | Declared loss behavior under a common logit shift | Causality or learned task performance |

Centered differences compare a derivative to the selected forward function; they cannot decide whether that function implements the intended causal task. Draw separate arrows for data dependence and gradient flow. For `logits[t]` predicting token `t+1`, observing input positions `0..t` is allowed; reading inputs beyond `t` is leakage.

## 3. Reproduce the mask-removal control

From foundations `labs/L1_autograd_mlp/`, in the full repository checkout with NumPy:

```sh
python decoder_bridge.py --output bridge-new.json
```

Use a new filename. The runner reuses four original foundations `Autograd` cases and all six architecture L1 cases; it does not rewrite their implementations or assertions. It also checks two smooth residual derivatives and the zero-branch identity. Then a temporary in-process patch makes the decoder's mask all-allowed, reruns two exact existing cases, and restores the original function. It requires exactly the intended assertion failure, no unrelated error, and a passing causality check after restoration. No source files are mutated.

[Actual NumPy report](../labs/L1_autograd_mlp/decoder_bridge_report.json): Python 3.12.14 / NumPy 2.3.5, seed-zero model, input `[3,14,15,9,26,5,35,8]`, changed suffix `[1,2,4]` at positions 5–7. Ten baseline cases pass. With the all-allowed decoder mask, the attention-definition case still passes but the exact original decoder-causality case fails. The maximum prefix-logit change is **0** with the causal mask and **1.7356425383462888** without it. Restoring the mask makes the existing causal check pass again.

The attention-definition case retains its own imported triangular mask: its independent operator-level assertion remains useful, while the decoder call site is broken. This is precisely why a correct attention formula does not qualify the integration.

The smooth residual fixture, seed 7 / float64 / centered step `1e-6`, has maximum sum-normalized relative errors about `6.58e-11` for the input and `2.85e-10` for the weight, below its fixture-specific `1e-6` threshold. Original MLP test tolerances remain unchanged. Hashes and the expected mutation failure are retained in the JSON; the expected failure is **not** a new failing baseline test.

## 4. Separate downstream PyTorch lane

The [selected training-check record](bridge_training_checks_2026_11.json) runs only `TestModel.test_causal` and `TestModel.test_z_loss` in a separate existing Python 3.12.14 / torch 2.9.1 CPU lane, with one torch thread and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Both pass. NumPy is absent in this lane and PyTorch emits an initialization warning; it is not a jointly installed NumPy/PyTorch environment or a fresh tested lock.

To repeat from training `labs/L1_pretraining_loop/` with the appropriate existing environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  test_pretraining.TestModel.test_causal test_pretraining.TestModel.test_z_loss -v
```

Read `train.batch` for the one-position target offset; this worksheet does not add a shifted-target test. The prior 600/1,500-step result tables, optimizer experiments and tested dependency pins are not rerun or updated. No model was newly trained for this bridge.

## 5. Submit predictions and limits

1. Derive the MLP parameter/gradient shapes and the residual identity contribution before running.
2. Predict which original tests catch a missing decoder mask, a wrong derivative, and a same-shape wrong operation. Explain why these are different failure classes.
3. Submit the complete baseline/mutant/restoration record, identify the exact expected failure, and explain the next-token shift separately.
4. Add an isolated negative control for shifted targets or document-boundary masking only after specifying its own reference. Existing prefix invariance does not cover those properties.

No learned quality, GPU performance, universal numerical tolerance or full-course qualification follows from these tiny CPU checks. Classroom comprehension remains untested.

Primary sources rechecked: [CS231n gradient checks](https://cs231n.github.io/neural-networks-3/) for centered differences and numerical limits; [Vaswani et al., arXiv:1706.03762](https://arxiv.org/html/1706.03762v7), §§3.1–3.3 for residual dimensions, positionwise FFNs, masking and the output offset. The shape/evidence diagrams are original course schematics, not copied artwork or paper result reproductions.
