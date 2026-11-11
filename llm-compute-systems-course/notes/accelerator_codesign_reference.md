# Accelerator co-design: specifications, code and worksheet

Companion to [deck 40](../slides/40_inference_accelerator_architectures.html) · [简体中文](zh/accelerator_codesign_reference.md). Optional reference, reviewed 2026-11-01. See the [evidence review](../references/accelerator_architectures_review.md) for primary sources, rejected media claims and roadmap provenance. No accelerator/compiler benchmark was run.

**Selected Google recheck, 2026-11-11:** The [new evidence ledger](../references/ironwood_spec_evidence_2026_11.md) supersedes the interpretation of the Google row: one chip has two separate memory spaces; a 4-chip VM exposes 8 JAX devices; usable HBM is below nominal capacity. The cause of the table/prose bandwidth difference is not established; rounding is not a verified explanation. The service log records preview 2025-11-24 and GA 2026-03-31. No local TPU/framework qualification ran. The original review date remains historical.

## Source-documented hardware limits

| Device | Documented fact | Boundary |
|---|---|---|
| Google TPU7x / Ironwood | Per chip: 2,307 BF16 TFLOP/s; 4,614 FP8 TFLOP/s; 192 GiB HBM; about 7.4 TB/s HBM bandwidth; 1,200 GB/s bidirectional ICI. Full pod: 9,216 chips. | [Google table](https://docs.cloud.google.com/tpu/docs/tpu7x); 7,380 GB/s in the table versus about 7.37 TB/s in prose motivates rounding. Peaks are not token throughput. |
| Huawei Ascend | Cube, Vector, Scalar and data-movement resources in the AI Core abstraction. | [Huawei documentation](https://www.hiascend.com/doc_center/source/en/canncommercial/800/opdevg/tbeaicpudevg/atlasopdev_10_0008.html); organization varies with product, so no universal core count or throughput is inferred. |
| Zhenwu 810E | A PPU-810E deployment choice is documented. | [Alibaba PAI](https://www.alibabacloud.com/help/en/pai/use-cases/one-click-deploy-deepseek); unverified numerical media specifications remain excluded. |
| Zhenwu M890 | 144 GB memory, 800 GB/s inter-chip bandwidth. | [Official May announcement](https://www.alibabacloud.com/en/press-room/alibaba-announces-comprehensive-full-stack-ai?_p_lc=1). Inter-chip bandwidth is not HBM bandwidth. |
| Zhenwu V900 | 216 GB memory, 1,200 GB/s inter-chip bandwidth; Q1 2027 production/commercial forecast. | [Official September announcement](https://www.alibabacloud.com/en/press-room/alibaba-unveils-roadmap-on-full-stack-ai-strategy). Announcement and forecast do not establish current capacity. |

Alibaba labels Zhenwu as its GPU family; the selected public material does not establish a specific systolic/SIMT PE organization. Vendor 3× statements are not a common disclosed cross-generation benchmark. Do not multiply them into an independently verified end-to-end claim. SAIL is an announced software stack; establish an actual repository, license, supported chip/SDK/driver pair and profiler before promising an integration. The source review records what was and was not established.

## Full single-tile teaching fragments

The lecture's short excerpts expose the abstraction. These expanded fragments still omit imports/decorators, launch, indexing, loop coverage, bounds/tail handling and runtime setup; **they are not standalone programs**. Assume valid full tiles, FP16 inputs/output and FP32 accumulation, with positive supported BM/BN/BK. Triton pointer tiles and TileLang tensor regions have already been formed correctly.

```python
# Triton: one K tile; a_ptrs/b_ptrs/c_ptrs are already formed.
acc = tl.zeros((BM, BN), tl.float32)
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
acc = tl.dot(a, b, acc)
tl.store(c_ptrs, acc.to(tl.float16))
```

```python
# TileLang: one K tile; correctly formed global regions are assumed.
f16, f32 = "float16", "float32"
shape_a, shape_b, shape_c = (BM, BK), (BK, BN), (BM, BN)
As = T.alloc_shared(shape_a, f16)
Bs = T.alloc_shared(shape_b, f16)
Cf = T.alloc_fragment(shape_c, f32)
T.clear(Cf)
T.copy(A_tile, As)
T.copy(B_tile, Bs)
T.gemm(As, Bs, Cf)
T.copy(Cf, C_tile)  # C_tile is an FP16 output region.
```

Read the [complete Triton GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) and [pinned TileLang GEMM example](https://github.com/tile-ai/tilelang/blob/e8225d7f5e8457c1360386e0603bc8196dcff095/examples/gemm/example_gemm.py) for the missing program structure. TileLang's [pinned language guide](https://github.com/tile-ai/tilelang/blob/e8225d7f5e8457c1360386e0603bc8196dcff095/docs/programming_guides/language_basics.md) documents shared/fragment scopes and manual async-copy wait obligations. Automatic layout/synchronization passes do not prove arbitrary code correct or establish support on every named chip. No fragments were compiled for this teaching edit.

## Analytical assumptions and experiment worksheet

For the array prediction, one useful row in a deliberately fixed 256-row tile gives 1/256 row occupancy. It is not measured overall chip utilization. For the output-stationary schedule, `t=k+i+j` assumes zero-based indices, skewed injection and unit-hop timing; the figure highlights only the k=0 wavefront at teaching tick 2. Other k streams are omitted. The ideal GEMM traffic bound assumes each input read once and output written once, with no spills; `sA/sB/sC` denote stored bytes per element.

Freeze the model/operator, shapes, precision, quality tolerance, latency target and total resources. Keep separate prefill and decode cases. Change one tile/layout/pipeline/quantization path, inspect generated code and resource use, then validate outputs before timing. Separate compile/startup, warm steady state, host transfers and communication. Attribute a gain to reuse, occupancy or removed overhead only with supporting evidence. Include a neutral/worse workload and an explicit unsupported-target outcome when no suitable device is available.

## Pinned full-kernel walkthrough — 2026-11-09

The [TileLang GEMM source-to-hardware supplement](../references/tilelang_gemm_walkthrough_2026_11.md) follows commit `72d986e1c45ba8e81f7611eb6835c9bece097d60`: launch, shared/fragment scopes, K pipeline, output conversion and remaining bounds/synchronization checks. Deck 40 slides 19/21 visualize it. The earlier single-tile fragments above remain conceptual; this adds a complete author-source reading cohort, not a newly compiled course kernel. The linked worksheet records full-tile, irregular M/N/K and small-M controls. Only integer coverage arithmetic ran; numerical/device validation and FA4/CuTeDSL remain pending.
