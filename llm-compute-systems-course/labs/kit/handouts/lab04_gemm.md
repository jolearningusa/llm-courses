# L4 — GEMM shapes, tiling and pipelines

**Released/validated:** CPU PyTorch shape study and numerical K-tiled reference in the L2 pinned environment. **Hardware extension:** CUDA/CUTLASS/cuBLASLt or AMD hipBLASLt/CK implementation study and benchmarking; not executed on the authoring host. Expected time: 5–8 hours plus environment setup, pending pilot.

## A. Reuse and arithmetic

Derive the input-only arithmetic intensity of a square T×T output tile across a K-step of T. Include the final output write and discuss partial accumulations. Explain how increasing T changes register/shared-memory needs and the number of independent tiles. Use the L0 calculator for whole-matrix work/traffic, not as a substitute for the tile-level model.

## B. Run the shape study

```sh
python3 gemm_lab.py --repeats 15 > gemm-cpu.json
python3 -m unittest discover -s tests -v
```

The code compares CPU `torch.mm`-equivalent execution with an explicit Python K-tile loop. Both validate against a higher-precision reference and include Python dispatch/checksum in wall timing. These small shapes illustrate prefill-like, decode-like and small-expert regimes; they do not reproduce a frontier model or use GPU Tensor Cores. Vary tile K and shape, inspect tolerance and identify overhead from extra dispatch and accumulation.

## C. Write or inspect a real kernel

On compatible hardware, implement a guarded shared-memory tiled GEMM or inspect a recorded CUTLASS/CK example. Identify how threads load tiles, synchronize, accumulate and store edge tiles. State dtype/accumulation/layout support. Validate irregular dimensions and memory/race behavior before performance work. The CPU K-loop is not the CUDA implementation: mapping each Python tile to a separate GPU matmul does not create the intended on-chip pipeline.

## D. Compare libraries and pipelines

Choose a supported cuBLASLt/CUTLASS or hipBLASLt/CK path and record version, algorithm/layout, workspace and actual backend. Match the mathematical contract and numerical threshold. Compare one tile/pipeline/epilogue choice at a time across all three regimes. Include kernel-only and allocation/copy-inclusive scope where relevant. Report first use, warm repetitions and an adverse case. Higher-level source syntax alone is not evidence of generated instructions or overlap.

Without device access, complete A/B and a source walkthrough for C/D, label hardware results unverified, and state the measurements needed to test the hypothesis. Use the [report template](../../report_template.md).

## Optimization-backend source comparison

Using [the technique map](../../../notes/optimization_techniques.md), select one supported library baseline and two eligible paths among Triton, TileLang, CuTe DSL and CUDA C++. Identify who controls tiles, layout, asynchronous movement and synchronization. Inspect an existing autotuner before proposing a new one. Record target/release constraints, correctness cases and the compile/search budget. This is a source worksheet unless you actually build/run the target; no TileLang or GPU validation is added by this documentation update.

<a id="pipeline-teaching-checkpoint"></a>
## Pipeline teaching checkpoint — deck 10

Allow 10–15 minutes (estimate, not classroom-piloted). Before running anything, draw four loads and four computes with two buffers. Assume independent copy/compute resources, load `L=3 μs`, compute `C=5 μs`, and no extra overhead. When is the last output ready? What prevents buffer reuse while a consumer is active?

From `labs/kit/`, run this standard-library calculation:

```sh
python3 - <<'PY'
n, load, compute = 4, 3, 5
serial = n * (load + compute)
pipeline = load + compute + (n - 1) * max(load, compute)
print('serial_us', serial, 'pipeline_us', pipeline)
print('speedup', round(serial / pipeline, 3))
PY
```

Expected: `serial_us 32 pipeline_us 23`, then `speedup 1.391`. Compute intervals are `[3,8]`, `[8,13]`, `[13,18]`, `[18,23]` μs. Loads are `[0,3]`, `[3,6]`, `[8,11]`, `[13,16]` μs. The second reuse of each buffer waits until its previous consumer finishes. The 5 μs steady-state period is not the finite four-tile latency. Repeat by hand with one tile (8 μs), then swap L/C (still 23 μs for four tiles under these assumptions).

Submit your prediction, annotated schedule, arithmetic output and one reason measured overlap may fail. For the C/D target extension, point to actual producer commit, consumer wait and consumer release operations at a pinned source revision. The [CUDA pipeline guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html) defines the producer/consumer protocol; this constructed arithmetic does not execute that API, model resource contention or measure GPU speed. More staging can consume resources and reduce occupancy; investigate rather than assume monotonic improvement.

中文提示：先画双缓冲的所有权，再计算四块的完成时间；23 μs 包含启动与排空，5 μs 只是稳态周期。提交预测、时序、输出及一个可能破坏重叠的条件。

## Optional pinned TileLang source dossier — 2026-11-09

Follow the [full-kernel worksheet](../../../references/tilelang_gemm_walkthrough_2026_11.md) with deck 40 slides 19/21. Submit (a) the fixed commit and BM/BN/BK contract, (b) launch/K-step coverage for full, irregular and small-M shapes, and (c) the exact generated-code, tail, synchronization and numerical evidence you would collect on an eligible target. Integer coverage alone does not execute TileLang. The CPU K-tiled helper's input-dtype accumulation must not silently substitute for the pinned kernel's FP32 fragment accumulation. Target compilation, safety/numerical checks and matched benchmarks are still optional **not-run hardware extensions** until actually performed.
