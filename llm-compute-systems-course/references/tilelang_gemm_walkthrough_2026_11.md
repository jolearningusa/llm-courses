# TileLang GEMM — pinned source-to-hardware walkthrough

Reviewed 2026-11-09 for deck 40 slides 19 and 21. This is the **TileLang source-reading portion of CMP-3**, not target qualification or the pending FlashAttention-4/CuTeDSL walkthrough. No TileLang package was installed, imported, compiled or executed. The only executed exercise was the standard-library integer coverage calculation below. Historical lab pins and the earlier conceptual Triton/TileLang comparison remain intact.

## Exact reading cohort

Upstream `main` was resolved once to **`72d986e1c45ba8e81f7611eb6835c9bece097d60`**. These immutable source links, rather than moving documentation, support the walkthrough. Files were downloaded and read as text, without executing upstream code.

| Pinned author source | Role in the reading |
|---|---|
| [GEMM example, lines 5–26](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/examples/gemm/example_gemm.py#L5-L26) | Tensor contract, launch, scopes, clear, K loop, copies and GEMM |
| [Example main, lines 29–57](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/examples/gemm/example_gemm.py#L29-L57) | One CUDA-tensor example, numerical assertion, generated-source accessor and profiler call; none run here |
| [Author example tests](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/examples/gemm/test_example_gemm.py) | `test_example_gemm` calls the example main; reading it is not running it |
| [Pipeline guide](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/docs/programming_guides/software_pipeline.md) | Inferred pipelines versus explicit stage/order annotations and dependency rules |
| [Language guide](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/docs/programming_guides/language_basics.md) | Tile scopes, layout responsibilities and the surrounding programming model |
| [Copy primitive implementation](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/tilelang/language/copy_op.py) | Ordinary tile copy versus explicitly asynchronous copy/wait ownership |
| [Safe-memory legalization pass](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/src/transform/legalize_safe_memory_access.cc) and [pass tests](https://github.com/tile-ai/tilelang/blob/72d986e1c45ba8e81f7611eb6835c9bece097d60/testing/python/transform/test_tilelang_transform_legalize_safe_memory_access.py) | Where to investigate generated bounds checks; pass presence is not proof this kernel's tails passed on a device |

Recorded SHA-256 for the two main reproducibility anchors:

- `examples/gemm/example_gemm.py`: `25d16f182bbe21d2cd0ea0edd9541b811120cba609c4d2c3aa4e800e8c946a99`.
- `docs/programming_guides/software_pipeline.md`: `0777f4e01f9ea28d7f257c222f6d044c23b20a4b148963bd7471aed124cc809d`.

Resolve neither source back to a new `main` during an experiment. Save the compiler build and submodule revisions as well: pinning this source does not install its dependencies or establish support for a particular accelerator. The moving [online pipeline guide](https://tilelang.com/programming_guides/software_pipeline.html) was also checked, but the links above define this reading cohort.

TileLang uses TVM-based lowering with explicit storage scopes and compiler-inferred layouts. Neither inference nor source availability establishes that this same program is supported or efficient on every advertised backend. Manual asynchronous operations retain explicit synchronization obligations.

## Follow one output tile

The example computes `C = A @ B` with A shaped M×K and B shaped K×N. Its defaults are FP16 inputs/output and an FP32 fragment accumulator. The author main chooses M=N=K=1024, BM=BN=128, BK=32, and 128 threads per launch block. It uses CUDA tensors; this main is not a tested CPU or Ascend entry point.

| Source operation | Meaning / question for generated code |
|---|---|
| Ceil-div launch over N and M | One block owns a logical BM×BN output tile; establish actual launch/target limits |
| Shared A and B allocation | BM×BK and BK×BN input tiles; layout inference can choose physical layout |
| Fragment C allocation and clear | BM×BN FP32 accumulation initialized once before the K reduction |
| Pipelined ceil-div K loop, stage count 3 | Two input copies followed by tile GEMM; inspect compiler-created buffer versions and dependency waits |
| GEMM on shared tiles | Reuses input tiles across multiple outputs; inspect target instructions, not just DSL names |
| Fragment-to-output copy | Conversion to FP16 output and the edge-store boundary; confirm in lowered IR/device code |

At the chosen sizes, A and B contain 8,192 bytes each and the logical FP32 accumulator contains 65,536 bytes. These are **shape-derived payloads**, not compiler resource reports. The fragment is distributed, not a 64 KiB allocation per thread. Pipeline buffering, shared-memory layout padding, registers/spills and occupancy must be read from generated artifacts and target tools. Do not infer a 16 KiB shared allocation or a 3× speedup from the source.

The regular pipeline asks the compiler to schedule copies and consumers. It does not demonstrate observed overlap. The pinned copy definition explicitly makes waits the programmer's responsibility for `T.async_copy`; replacing ordinary copies with manual async operations changes the synchronization contract. Record commit/wait/reuse dependencies before calling such a rewrite equivalent.

## Executed exercise: coverage arithmetic only

Keep BM/BN/BK at 128/128/32. Grid dimensions below are `(ceil(N/BN), ceil(M/BM))`; K steps are `ceil(K/BK)`.

| M / N / K | Output grid | Output tiles | K steps | Control |
|---|---|---|---|---|
| 1024 / 1024 / 1024 | 8×8 | 64 | 32 | Full-tile author example |
| 130 / 129 / 33 | 2×2 | 4 | 2 | M, N and K all have tails |
| 1 / 128 / 32 | 1×1 | 1 | 1 | Small M, no opportunity for a long K pipeline |

```python
from math import ceil
for m, n, k in [(1024, 1024, 1024), (130, 129, 33), (1, 128, 32)]:
    gx, gy, steps = ceil(n / 128), ceil(m / 128), ceil(k / 32)
    print((m, n, k), (gx, gy), gx * gy, steps)
```

This snippet was run on Python 3.9.6 using only the standard library, giving exactly the three rows above. It executes no tensors, compiler or target kernel. Tile coverage is not physical occupancy or delivered utilization. For the irregular case, inspect the final K tile: only reduction index 32 is valid, while the other 31 positions must contribute zero. Extra output coordinates must not write past M/N. Ceil-div alone proves neither condition.

## Device worksheet: what is still missing

1. **Freeze the lane.** Record the supported GPU, driver/runtime, Python, exact TileLang build/submodules, pass options, dtype/layout, tile sizes and model/operator inputs. Retain the baseline generated IR/device source with its digest and resource report. The pinned example's `get_kernel_source()` is a starting point, not evidence that we generated code here.
2. **Inspect tails and synchronization.** With safe-memory legalization enabled, trace A/B bounds predicates, invalid-K zero fill and output-store predicates. Inspect the actual pipeline waits and buffer reuse. Compare with the pinned safe-memory tests; run selected upstream tests and target memory/race tooling, recording tool coverage. Compile rejection, unsupported target, test failure and not-run are separate outcomes.
3. **Use an independent numerical oracle.** Convert the same quantized FP16 inputs to FP64 for reference multiplication, retaining the pre-conversion reference and its FP16-rounded form. Test zeros, signed basis inputs, cancellation/large-dynamic-range values within a declared finite range and seeded random inputs across all three shapes. Reject unexpected NaN/Inf. Predeclare scale-aware absolute/relative error and conversion policy; report max absolute and relative error with a denominator floor. The author's example uses `rtol=atol=1e-2` for one random shape, not a universal correctness requirement. For exact zeros/basis cases, use literal expected outputs too.
4. **Avoid an accidental reference mismatch.** L4's existing `tiled_gemm` keeps the output in the input dtype and adds K-chunk products into it. Simply calling it with FP16 does not reproduce this kernel's FP32-across-K accumulation. It remains useful for CPU shape teaching, not as an automatically equivalent low-precision oracle.
5. **Measure one change only after correctness.** Compare eligible pipeline depths with everything else fixed; compile each variant separately and recheck outputs/tails. Report compiler/startup cost, cold and warm device latency and transfer-inclusive scope separately. Inspect registers/shared memory and a device timeline before claiming overlap. Include small M, irregular shapes and neutral/worse outcomes; fix search budget and record failed candidates. A matched supported library baseline needs the same contract and timing boundary.

No generated code, target compilation, numerical tensor comparison, sanitizer result, device trace or latency was produced in this update. The exact irregular shapes are proposed tests, not newly supported shape claims. The FlashAttention-4 half of CMP-3, backend portability and device benchmarks remain open.
