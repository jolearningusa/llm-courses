# AMD ROCm: a second implementation of the systems stack

AMD provides an open-source compute software ecosystem, **ROCm**, with libraries for training and inference. The closest response to “AMD's open library” is the ecosystem as a whole, especially **AITER** for AI operators and **Composable Kernel** for kernel construction. These layers have different responsibilities.

## Compare responsibilities, not identical implementations

| Responsibility | NVIDIA-side teaching target | AMD-side teaching target | Primary AMD source |
|---|---|---|---|
| GPU programming/runtime | CUDA | HIP | [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| BLAS and flexible GEMM | cuBLAS / cuBLASLt | rocBLAS / hipBLASLt | [rocBLAS](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/), [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) |
| Composable kernel construction | CUTLASS / CuTe | Composable Kernel (CK) | [Current library repository](https://github.com/ROCm/rocm-libraries), [CK documentation/mirror](https://github.com/ROCm/composable_kernel) |
| Optimized AI operators | TE, attention/MoE libraries, specialized runtime kernels | AITER, with multiple kernel backends | [AITER](https://github.com/ROCm/aiter) |
| Multi-GPU collectives | NCCL | RCCL | [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/) |
| GPU-initiated one-sided communication | NVSHMEM | rocSHMEM | [rocSHMEM](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/) |
| Timeline and per-kernel profiling | Nsight Systems / Compute | ROCm Systems Profiler / Compute Profiler, ROCprofiler-SDK | [Profiling overview](https://rocm.docs.amd.com/en/latest/components/profiling-tools-overview.html) |
| Framework/compiler path | PyTorch + NVIDIA backend, Triton language | PyTorch + ROCm backend, supported Triton paths | [PyTorch compatibility](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html) |

This table maps tasks, not feature parity or drop-in API equivalence. For example, AITER spans optimized inference and training operators using Triton, CK and assembly backends; it is not a one-to-one replacement for Transformer Engine's precision recipe integration. A common PyTorch source program may select a different backend and require different kernels, layouts or tuning.

The source check found that the older `ROCm/composable_kernel` repository identifies itself as deprecated/read-only and points to `ROCm/rocm-libraries`. hipBLASLt documentation also points to the library monorepository. Use recorded commits from current upstream locations, and document migrations rather than cloning an old tutorial's paths blindly. RCCL documentation points to `ROCm/rocm-systems` for its current source.

## Hardware concepts to compare

Use a supported AMD Instinct device and its architecture documentation as the second hardware example. Compare execution groups, register/LDS usage, matrix instructions, memory hierarchy and interconnect topology against the NVIDIA workload analysis. Do not translate a warp-size assumption or a Tensor Core/TMA idiom mechanically. Wavefront behavior, compiler lowering and architecture support must be checked for the actual device.

HIP source portability, functional portability and performance portability are separate claims. Shared high-level APIs reduce source changes; they do not guarantee identical launch choices, numerical behavior, supported precision or performance. Likewise, a shared dtype name is insufficient to establish identical encoding, scaling recipe or fast hardware path.

## Course integration

In weeks 3–7, run the same tensor workload through PyTorch and the selected kernel path where hardware permits. In week 13, compare collective latency/bandwidth with NCCL/RCCL under documented topologies. In weeks 8–10, compare MoE operator dispatch and serving backends using the same checkpoint and requests when both support them. In week 12, present a portability dossier identifying what was reused, retuned, replaced or unsupported.

The hands-on AMD track is optional when no AMD hardware is available; the library comparison and code-reading exercise are core content. Without both vendors' hardware, analyze supplied traces or public experiments with their limitations and make no local cross-vendor speed claim.

## Matched-workload assignment

Use one normalization/reduction, three GEMM shapes (training/prefill, decode, grouped MoE), and one supported attention configuration. Record operation semantics, dimensions, dtype/scaling, mask, batch/context distributions, software commits, actual backend and hardware. Validate output error before timing. Report compilation, warm execution, peak memory and achieved bandwidth/compute separately.

Then examine a training step or serving request containing that operation. Explain whether a local kernel improvement remains material after communication, input, runtime and queueing costs. Compare percent-of-achievable-performance within each system before making an absolute speed or cost claim. Disclose power boundaries and resource costs when presenting efficiency comparisons.

This exercise teaches how to build a portable system with explicit specialization points, rather than how to repeat a vendor benchmark.

## 2026-10-27 operator-library refresh

The current AITER README additionally describes FlyDSL/OPUS and Iris-based communication paths. Treat these as backend-specific extensions and inspect target/operator coverage; a single AITER installation does not establish every backend is available. The core responsibility map remains HIP/runtime, GEMM/construction, AI operators and communication. No AMD kernel was executed on the authoring host.

## Programming-model comparison (2026-10-28)

[Deck 33](../slides/33_programming_models_beyond_cuda.html) places HIP next to the other vendors' answers to CUDA (SYCL, Metal, Pallas, NKI) and the portable tile DSLs (Triton, TileLang). It records the HIP execution-group width (64 on CDNA, 32 on RDNA), AMD's layered kernel options (HIP, CK Tile, FlyDSL, Triton, TileLang, AITER) and the three routes for running CUDA code on AMD (HIPIFY, SCALE, ZLUDA). Sources are in the [bibliography](../references/bibliography_and_links.md#programming-models-beyond-cuda-reviewed-2026-10-28).
