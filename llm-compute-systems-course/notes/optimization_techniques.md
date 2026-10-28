# Optimization techniques and decision systems

Reviewed 2026-10-27. This chapter integrates relevant ideas from the user-provided shared discussion, checked against primary sources. The shared chat is a lead, not technical authority. No new GPU, serving-engine or edge performance result is claimed.

## The execution alternatives

An application may use an optimized library, graph compiler or custom kernel for different regions. These are choices at related layers, not a mandatory chain `PyTorch → cuBLAS/cuDNN → TileLang → arbitrary hardware`. CUDA supplies a platform/toolchain; cuBLAS/cuDNN supply optimized operations; NCCL supplies collectives; TensorRT supplies a model optimization/execution engine. Triton and TileLang supply kernel programming/compiler abstractions. A system can combine supported paths while preserving tensor, numerical and runtime contracts.

TileLang is distinct from **CUDA Tile**, **CuTe DSL**, and DeepSeek **TileKernels**. The latter is a library written using TileLang. Do not rank these by stars as if popularity proved coverage or performance. The course retains CUDA fundamentals and adds a comparison of abstractions, generated code, layout and synchronization.

Primary sources: [CUDA documentation](https://docs.nvidia.com/cuda/), [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html), [TileLang](https://github.com/tile-ai/tilelang), [TileLang tools](https://www.tilelang.com/tools/index.html), [TileKernels](https://github.com/deepseek-ai/TileKernels), [Torch-TensorRT](https://docs.pytorch.org/TensorRT/user_guide/torch_tensorrt_explained.html).

## Technique-to-component map

| Bottleneck / requirement | Candidate techniques | Component to inspect | Required evidence / adverse case | Course practice |
|---|---|---|---|---|
| CPU starvation / transfers | NUMA locality, bounded prefetch, pinning, pipelined copies | Input worker, allocator, runtime/OS | End-to-end trace; contention and buffer lifetime | L1 |
| Launch / intermediate traffic | Fusion, graph capture/replay, compiler region selection | Dynamo/Inductor, runtime, custom op | Actual graph breaks, allocations and launch trace; small/dynamic shapes | L2–L3 |
| Matrix throughput / reuse | Library algorithm, tiling, layout, Tensor Core path, async staging | cuBLASLt/CUTLASS, Triton/TileLang/CuTe, AMD libraries | Matched GEMM/accumulator, tails, resource use and conversion costs | L4 |
| Attention traffic / stage balance | Online normalization, exact tiling, pipeline partitioning | SDPA/FlashAttention/FlashInfer or target backend | Same mask/positions/dtype, output/gradients; dense work remains quadratic | L5 |
| Precision / capacity | AMP, block scaling, PTQ/QAT, packing | TorchAO, Transformer Engine, ModelOpt and target kernels | Held-out quality, calibration split, scale/padding metadata and fallback | L6 |
| Training memory / communication | DDP/FSDP/ZeRO, TP/PP/CP, checkpointed activations, overlap | Training stack, mesh/groups, collective library | Fixed effective batch/update progression, peak memory and exposed waits | L7–L8 |
| MoE irregularity | Permute/group/dispatch/combine, expert placement, skew mitigation | Dispatcher, grouped GEMM, DeepEP / target stack | Matched routing/drop semantics; empty/hot experts and backward | L9 |
| Lost work | Consistent snapshot, async staging, restart state and useful-throughput accounting | Checkpointer, optimizer/RNG/data state | Interrupted/replayed trajectory; incomplete snapshot and completion boundary | L10 |
| Repeated prefix work | Compatible exact-token prefix reuse, locality and eviction policy | vLLM/SGLang/llama.cpp state manager | Ownership/invalidation, pressure, matched tokens; distinct from semantic responses | L11 |
| Queue/phase interference | Chunked prefill, admission, phase placement, compatible KV handoff | Scheduler, router, connector | Equal resource budget, arrival replay, failures and SLO goodput | L12 |
| Long tool waits / draft overhead | Lifecycle hints, retain/offload/recompute, adaptive speculation | Agent runtime + serving policy | Task latency, byte-seconds, exact verification, acceptance/load regimes | Planned L13 |
| Deployment / cold start | Export contracts, AOT packaging, delegation and memory planning | AOTInductor, Torch-TensorRT, ExecuTorch / target SDK | Producer/loader/device/version compatibility; unsupported shape and partition boundaries | Planned L14 |
| Automated optimization | Diagnose → select layer → bounded search → validate → record / stop | Controller separate from evaluator | Search cost as well as final performance; held-out traces and negative transfer | Supplementary deck 32 |

These candidates are hypotheses until measured. Low aggregate GPU utilization alone does not identify which row applies.

## Existing tools are baselines

- **vLLM auto-tuning:** the inspected [README](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md) already searches `max-num-seqs` and `max-num-batched-tokens` with optional constraints. A proposed controller should compare with this and simple random/grid search under the same budget.
- **TileLang autotuning:** the [autotuning guide](https://www.tilelang.com/programming_guides/autotuning.html) and [tool index](https://www.tilelang.com/tools/index.html) provide tuning/debugging/analysis entry points. Reusing these is a baseline, not evidence of a universal PyTorch converter. Analytical estimates do not replace device timing.
- **FlashInfer-Bench:** [paper, arXiv:2601.00227](https://arxiv.org/abs/2601.00227), [project](https://bench.flashinfer.ai/), and [maintainer description](https://flashinfer.ai/2025/10/21/flashinfer-bench.html) describe a workload/evaluation/substitution loop. A new project needs a testable distinction such as choosing the optimization layer or reducing total search cost.
- **Related survey:** [Towards Automated Kernel Generation in the Era of LLMs, arXiv:2601.15727](https://arxiv.org/abs/2601.15727). Read for a taxonomy and benchmark leads; this review checked the abstract/scope, not every surveyed implementation.

A useful proposed controller has separate planning and execution/evaluation components. The evaluator enforces input/semantic constraints, numerical/task-quality thresholds, resource/time limits, reproducible timing and an immutable record of success **and failure**. The planner may rank old candidates, but must revalidate each new target. Combining two accepted changes creates another candidate requiring another evaluation.

## Clarify export and “AOI” terminology

“AOI” does not identify a standard artifact in the reviewed stack. The shared discussion suggests several possible intended tools. Keep them separate:

| Path | Purpose / output boundary |
|---|---|
| AOTInductor | Compile a torch.export graph ahead of deployment; PT2 package and compatible loader |
| Torch-TensorRT / TensorRT | Optimize supported graph regions for NVIDIA execution, with configuration-dependent fallback and packaging |
| ExecuTorch | Target lowering plus lightweight runtime; `.pte` and optional separate `.ptd` data/backend payloads |

The [ExecuTorch CUDA backend documentation](https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html) describes AOTInductor/Triton integration. That is one backend, not a statement that every ExecuTorch target runs CUDA or produces identical kernels. [Torch-TensorRT AOT deployment](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/aot_inductor.html) also documents a composition path. Check selected release/device constraints rather than merging these product names into one converter.

## DeepInfer and service integration

The inspected [DeepInfer README](https://github.com/deepseek-ai/DeepInfer) lists a specialized offline release, model/target requirements and external service components still required. It is not a general laptop/AMD/Rubin compatibility claim. Its “KV cache system” service gap must not be paraphrased as proof that its decoder computes without any internal KV state. Service APIs, state ownership and integration feasibility need source-level investigation before proposing an adapter.

Pinned inspected snapshots:

| Repository | Commit |
|---|---|
| DeepInfer | `553741557a57df44a46fb3ab72b6aafeaf945337` |
| TileKernels | `c1cf914a82e6901c679445f872051ce44ac2cc63` |
| TileLang | `0408712ead01b4383e0f63e9e0bf74c35c1d7676` |

Review depth: README/tool documentation and repository identity/support scope, not installation, full source audit or reproduced performance.

## Agent hints: proposal versus implementation

[vLLM issue #52113](https://github.com/vllm-project/vllm/issues/52113), “Session-Aware KV Cache Hints for Agentic Workloads,” was **open and labeled RFC** when checked on 2026-10-27. It motivates bounded lifecycle intent for tool waits/branches. This is proposal evidence, not a shipped API guarantee. [agentic-api](https://github.com/vllm-project/agentic-api) already addresses stateful agent serving; a new gateway must identify an unmet requirement. Applications should not confuse session identity or TTL preference with exact compatible token identity or ownership of physical engine pointers.

## Cross-vendor and edge extension

[ExecuTorch backends](https://docs.pytorch.org/executorch/stable/backends-section.html), [Qualcomm AI Hub](https://aihub.qualcomm.com/), [Core ML Tools](https://github.com/apple/coremltools), [OpenVINO](https://docs.openvino.ai/), [Ryzen AI](https://ryzenai.docs.amd.com/en/latest/) and [LiteRT](https://ai.google.dev/edge/litert) are useful target-specific entry points. Compare support, partitioning, data movement and whole-application quality/performance. They are not promised interchangeable backends. The course remains centered on LLM systems; medical-image deployment from the shared discussion is an optional separate domain-transfer project, requiring domain-specific evaluation rather than a clinical claim.

Coverage and implementation limits are recorded in the [component audit](course_component_audit.md).
