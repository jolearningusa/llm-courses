# FlashAttention-4 SM100 — pinned pipeline and qualification worksheet

Reviewed 2026-11-09. Companion to deck 12 slides 15/17 and L5. This completes the **FA4 source-reading supplement** proposed in CMP-3, alongside the [TileLang source walkthrough](tilelang_gemm_walkthrough_2026_11.md). Generated-code/device correctness, performance and cross-backend evidence remain open. No FA4 package, model, CUDA build or upstream test was installed or run in this update.

## Fixed author-source cohort

Resolved upstream main once to `c055bdc6444a1a1984082a64b7fd62d6cb42a93d`. These files were downloaded and read as text, without importing them:

| Pinned source | What to inspect |
|---|---|
| [CuTeDSL README](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/README.md) | Advertised Hopper/Blackwell implementation and public entry points; not a tested environment here |
| [Host interface](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/interface.py#L627-L672) | Dtype/device/head-count checks, head-dimension validation and downstream dispatch; public package name is not a kernel identity |
| [Ordinary SM100 dispatch](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/interface.py#L1347-L1369) | Class/config selection, including a separate SwapAB path; record actual selected class and arguments |
| [SM100 forward kernel](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/flash_fwd_sm100.py#L871-L929) | Specialized roles; read `load`, `mma`, `softmax`, correction and epilogue methods within the same file |
| [S/P/O pipeline initialization](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/flash_fwd_sm100.py#L1038-L1068) | Two-way readiness rather than a simple unidirectional queue |
| [Output rescale](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/flash_fwd_sm100.py#L3139-L3187) | TMEM load, scale, store and asynchronous-view fence for old partial output |
| [Softmax state](https://github.com/Dao-AILab/flash-attention/blob/c055bdc6444a1a1984082a64b7fd62d6cb42a93d/flash_attn/cute/softmax.py#L305-L402) | Row maximum/sum updates, base-2 scaling and optional rescale-threshold behavior |

SHA-256 anchors:

- `flash_attn/cute/interface.py`: `588c82672d27010e76d3b5f18b93648819988e0ea3bb838177e41c8161df8ebc`.
- `flash_attn/cute/flash_fwd_sm100.py`: `c89147dcbb8f8ce17285ffedbcc9097968ce0adb4bf381de7b91d75f2e088abe`.
- `flash_attn/cute/softmax.py`: `aaddf3030dba7b9809c4fefae3caab22d0f22b9ecf5050601d6646aefb611a49`.

The earlier [FA4 paper](https://arxiv.org/abs/2603.05451) remains the conceptual reading on asymmetric scaling. This source cohort is a later implementation snapshot, not a claim to reproduce the paper's exact evaluated binary or configurations. The interface also contains other architecture/MLA/decode branches; neither the README nor one source file establishes that all combinations are supported. We inspect the ordinary SM100 forward path only.

## Read the dependencies, not just the MMA calls

1. **Load:** transfer Q/K/V tiles into shared storage. The chosen implementation has TMA and alternative paths controlled by configuration; save the actual path and layout rather than inferring it from the library name.
2. **QK:** the MMA role produces a score tile S and signals that softmax may consume it.
3. **Softmax and correction:** create probability tile P and updated row statistics, while rescaling the accumulated old O for the changed normalization reference. These are distinct dependencies.
4. **PV:** the MMA role must wait for P **and** old-O correction readiness before adding the current P×V contribution. QK and PV in the drawing reuse the same MMA role; they are not a claim of two independent MMA warps.
5. **Epilogue:** final normalization/output conversion/store. Trace the final-write and reuse boundaries separately from the steady-state tile loop.

The source's `pipeline_s_p_o` comment explicitly describes this bidirectional exchange: MMA signals S readiness; softmax/correction signal P/old-O readiness back to MMA. Thus a method named `consumer_release` can carry more meaning than “storage is now free.” Follow the particular stage index, phase and participating groups. Additional pipelines handle final P splits, O readiness, row statistics and epilogue handoff. The figure intentionally omits their detailed scheduling and resource counts.

A plausible profiling hypothesis is that faster matrix operations expose softmax, rescaling, movement or readiness waits. To establish that on a device, collect the selected configuration, generated instructions/resources and a timeline/counter signature. A source dependency graph is not a measured overlap trace or evidence that any specific stage is currently the bottleneck.

## Executed scalar checkpoint: both old state terms must rescale

For scores `[0, ln 2]` and scalar values `[2, 5]`, processing one score per tile gives output **4**. After the first tile: maximum m=0, denominator l=1 and unnormalized weighted sum o=2. After the second tile, the new maximum is ln 2 and the old-state factor is α=1/2. Then l=1/2+1=1.5, o=2×1/2+5=6, and o/l=4.

```python
import math
alpha = math.exp(0.0 - math.log(2.0))
denominator = 1.0 * alpha + 1.0
weighted = 2.0 * alpha + 5.0
correct = weighted / denominator
missing_old_output_rescale = (2.0 + 5.0) / denominator
missing_old_denominator_rescale = weighted / (1.0 + 1.0)
assert abs(correct - 4.0) < 1e-12
assert abs(missing_old_output_rescale - correct) > 0.5
assert abs(missing_old_denominator_rescale - correct) > 0.5
print(correct, missing_old_output_rescale, missing_old_denominator_rescale)
```

Executed on Python 3.9.6 standard library: `4.0 4.666666666666667 3.0`. The two deliberate omissions fail the literal expected result. This checks a teaching identity, **not** FA4 code, synchronization or tensor numerics. SM100 uses base-2-scaled state and configurable rescale behavior; this real-arithmetic illustration is not a line-by-line or bit-exact model of those implementations. L5's existing CPU reference likewise does not execute the author kernel.

## Concrete target worksheet — not run

Start with candidate dense self-attention: batch 1; Q/K/V heads 8/8/8; Q/K and V head dimensions 64/64; matching contiguous FP16 or BF16 CUDA inputs; equal Q/K sequence lengths N=128,129,1; explicit softmax scale 1/8; causal and noncausal cases separately. The host checks matching input dtype, CUDA placement outside fake mode, head-count divisibility and architecture-dependent dimensions/alignment. Choosing this candidate does not establish that its install/compile/dispatch succeeded, and an N tail depends on the actual selected tile size.

- **Freeze:** pin package/source/build dependencies, GPU architecture, driver/CUDA, input revision, scale, mask semantics, pass/config options and environment variables. Save the actual selected class (ordinary SM100 versus SwapAB/MLA/other target), tile sizes, stages and compilation cache state. Do not bypass the device or architecture checks to report target support.
- **Correctness:** compare an independent dense FP64 oracle constructed from the same quantized inputs, preserving final conversion policy. Predeclare absolute/relative error and denominator floor; reject unexpected NaN/Inf. Use zeros, signed basis values, seeded random values and declared scale ranges. Inspect and test query/key tails and causal future-leakage controls. Non-square decode offsets, GQA, paged/ragged KV, arbitrary or all-masked rows, low precision variants and backward each need separate contracts and tests.
- **Safety and generated code:** inspect actual bounds/masking paths, pipeline stage/phase signals, TMEM/shared lifetimes and final output stores. Retain generated IR/device code, compiler resource reports and applicable device memory/race-tool output. This update produced none of these; scalar checks cannot replace them.
- **Performance:** only after correctness, compare matched eligible baselines, retaining compilation, cold/warm and end-to-end scopes separately. Fix the search budget and include a neutral/worse case. Use measured waits, traffic and resource evidence to test the bottleneck hypothesis; do not infer speed from source syntax, a vendor peak or the paper's historical chart.

L5's implemented reference is CPU FP32/64, single-head, nonempty square self-attention with causal/noncausal masks. It does not implement these multi-head CUDA candidates, arbitrary masks, all-masked rows, GQA/MLA, ragged sequences or offset decode. No new target support, model-quality, memory or latency result is claimed. CMP-3's two source-reading supplements are now available; target qualification and cross-backend comparison remain pending.
