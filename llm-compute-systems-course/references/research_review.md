# Research review and course design decisions

Prepared 2026-10-26 from primary papers, official documentation, university course pages and upstream GitHub repositories. This review precedes executable course implementation. It is a curated engineering review, not a claim of exhaustive literature coverage or independent reproduction of published results.

## Scope and review depth

The organizing question is **how to build an efficient computing stack for LLM training and inference**. Modeling appears where it changes workload shape, state, numerical requirements or communication. Research was expanded from serving to distributed training, PyTorch deployment, NVIDIA's hardware/software stack and AMD ROCm in response to the course requirements.

Three evidence levels are distinguished:

- **Selected-section review:** design/evaluation sections of FlashAttention-4, DistServe and FlashInfer; NVIDIA architecture and library documentation; PyTorch compilation/deployment and AMD library documentation. These ground mechanism and limitation discussions.
- **Bibliographic/abstract review:** foundational training, inference and architecture papers listed in the bibliography. Titles and central contributions were checked; this does not constitute line-by-line review of all proofs, code or experiments. Each teaching module must deepen the assigned readings before release.
- **Repository/interface review:** upstream READMEs, documented features, support information and source locations. No upstream project was installed or benchmarked as part of the literature review; exact code paths and compatible commits remain lab-release work.

Search results and discussion sites helped discover material. Technical claims in the course rely on the linked primary sources. Sources with failed or moved URLs were followed to official replacements where possible. The bibliography distinguishes required foundations, advanced readings and implementation references.

## Existing courses and tutorials

| Resource | What it contributes | Decision for this course |
|---|---|---|
| [Stanford CS349E](https://cs349e.github.io/) | Explicitly combines inference, hardware/performance modeling, kernels, compilers and disaggregated systems | Strong structural reference for serving; add a substantial training/recovery block and PyTorch deployment path |
| [Stanford CS336, Spring 2025](https://cs336.stanford.edu/spring2025/) | Build-and-measure language modeling systems; assignment 2 includes profiling and Triton attention | Use the systems assignment style, while bounding model implementation and training compute |
| [How To Scale Your Model: Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | Compute, data movement and capacity models across devices | Make prediction before measurement a recurring requirement; distinguish peak and achieved rates |
| [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | Progressive executable examples from vector addition to softmax, matmul and attention | Use as an alternative implementation path after explaining CUDA execution and memory |
| [GPU MODE lecture materials](https://github.com/gpu-mode/lectures) | Practical kernel-engineering talks and companion material | Supplement targeted lectures rather than require an unbounded video list |
| [NVIDIA TileGym tutorials](https://github.com/NVIDIA/TileGym) | Current tile-programming lessons and exercises | Advanced comparison of abstraction/toolchain tradeoffs; do not let it replace SIMT fundamentals |

The gap addressed here is the continuous path from framework semantics to kernels, communication, host/OS behavior and whole-job objectives, with both training and inference. This is a curriculum design judgment rather than a claim that no other course covers it.

## Paper synthesis: mechanisms and the experiments they imply

| Research thread | Primary anchors | Engineering lesson | Experiment or critique |
|---|---|---|---|
| IO-aware attention | [FA1](https://arxiv.org/abs/2205.14135), [FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451) | Algorithm, partitioning and hardware pipeline evolve together | Compare identical semantics/shapes and determine whether memory, softmax, scheduling or matmul limits each kernel |
| Dynamic attention runtime | [FlashInfer](https://arxiv.org/abs/2501.01005) | Runtime planning and kernel scheduling must handle changing sequence lengths | Vary raggedness at constant total tokens; inspect planner, workspace and replay costs |
| Training state | [ZeRO](https://arxiv.org/abs/1910.02054), [FSDP](https://arxiv.org/abs/2304.11277) | Capacity savings introduce communication/lifetime tradeoffs | Predict state memory, then compare peak allocation and all-gather/reduce-scatter timelines |
| Training parallelism | [Megatron](https://arxiv.org/abs/2104.04473), [selective recomputation](https://arxiv.org/abs/2205.05198) | Layout, parallelism and recomputation interact | Keep effective batch fixed; explain memory saved, communication exposed and pipeline bubbles |
| Serving scheduling/cache | [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [PagedAttention](https://arxiv.org/abs/2309.06180), [SGLang](https://arxiv.org/abs/2312.07104) | Request iteration and shared state matter alongside a fast forward pass | Measure cache reuse, allocation and queueing, with cancellation/eviction behavior |
| Prefill/decode separation | [Sarathi-Serve](https://arxiv.org/abs/2403.02310), [DistServe](https://arxiv.org/abs/2401.09670), [Mooncake](https://arxiv.org/abs/2407.00079) | Compute interference and KV transport compete | Sweep load, prompt length and transfer bandwidth under fixed total GPU resources |
| Low precision | [SmoothQuant](https://arxiv.org/abs/2211.10438), [AWQ](https://arxiv.org/abs/2306.00978), [TorchAO](https://arxiv.org/abs/2507.16099) | Storage savings need supported kernels and numerical evaluation | Include scale/packing costs and quality/loss checks; measure adverse shapes |
| Speculation | [Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318) | Parallel verification can reduce serial target steps | Sweep draft cost/acceptance/load and use correct distribution-preserving correction |
| Architecture as workload | [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | Heads, compressed state and expert routing alter bottlenecks | Brief primer, then use the actual tensor/state/traffic shape in system experiments |

### Three deeper findings that affect teaching

**Historical kernel baselines age quickly.** FlashAttention-4's evaluation specifies B200, dtype, head dimension and library baselines, and notes that newer cuDNN versions incorporated techniques that yield similar performance. The teaching exercise should compare current compatible backends and causal mechanisms, not promise an old headline speedup. Its scheduling section also connects tile ordering with load balance and L2 locality. [FA4 design/evaluation](https://arxiv.org/html/2603.05451v1).

**Disaggregation is conditional.** DistServe's evaluation used 32 A100-80GB GPUs, a specified inter-node bandwidth, OPT workloads and explicit latency objectives with generated arrivals. Its placement choices reflect those constraints. Change attention state size, network, load or model and the transfer/interference balance changes; a single-GPU simulation cannot establish multi-node speedup. [DistServe evaluation](https://arxiv.org/html/2401.09670v3).

**A fast attention kernel still needs a runtime.** FlashInfer's design separates storage formats, compute abstraction and dynamism-aware planning. Its schedule maps variable work to CTAs and produces deterministic aggregation for a fixed shape configuration; workspace and CUDA Graph compatibility enter the design. Teach planning time and kernel execution together. [FlashInfer runtime](https://arxiv.org/html/2501.01005v2).

## Implementation targets and why they were selected

Use [TorchTitan](https://github.com/pytorch/torchtitan) for a readable PyTorch-native distributed training path and [Megatron-LM/Core](https://github.com/NVIDIA/Megatron-LM) for a complementary NVIDIA-focused implementation. [llm.c](https://github.com/karpathy/llm.c) helps expose a small training execution path; [llama2.c](https://github.com/karpathy/llama2.c) is an inference code-reading option. These are not equivalent production platforms.

For serving, begin with [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), then follow a real request through [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang). Compare [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) only on a supported matched configuration. Use [Dynamo](https://github.com/ai-dynamo/dynamo), [NIXL](https://github.com/ai-dynamo/nixl), [Mooncake](https://github.com/kvcache-ai/Mooncake) or [LMCache](https://github.com/LMCache/LMCache) when studying routing/transfers/cache tiers, rather than installing all of them for a single lab.

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and [DeepEP](https://github.com/deepseek-ai/DeepEP) connect MoE numerical execution and communication. [CUTLASS](https://github.com/NVIDIA/cutlass), [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer) expose specialized implementation choices. Select one inspectable code path per assignment and record its commit.

## PyTorch deployment findings

Compilation, export and quantization are different operations. The [PyTorch stack note](../notes/pytorch_stack.md) follows eager/autograd, Dynamo/Inductor, AOTInductor and ExecuTorch, including the ambiguity of `.pt2` filenames. AOTI likely resolves the requested “AOI” terminology, but that remains an explicit interpretation. ExecuTorch's edge program and optional separate tensor data are distinct from an AOTInductor package. Deployment comparisons must include build/load/first-use and steady-state behavior.

## Hardware and vendor-library findings

[Rubin](../notes/vera_rubin.md) is useful because its hardware description links memory, matrix execution, data movement and communication. Keep the source-documented NVL72 configuration separate from later POD extensions and distinguish published limits from measured workload rates. [NVIDIA's library map](../notes/nvidia_stack.md) assigns kernels, precision, communication, transfer and runtime responsibilities.

The [AMD comparison](../notes/amd_stack.md) makes portability explicit. [AITER](https://github.com/ROCm/aiter) documents multiple kernel backends and both training/inference operators; [CK](https://github.com/ROCm/composable_kernel) and hipBLASLt source locations have moved into [rocm-libraries](https://github.com/ROCm/rocm-libraries). Do not copy stale repository paths from a tutorial. HIP, BLAS, collectives and profilers should be compared by responsibility and measured workload.

Another freshness issue: the old `NVIDIA/NeMo` URL now redirects to Speech. Current LLM training material should use [NeMo's current documentation](https://docs.nvidia.com/nemo/) and [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html). This is a concrete reason to verify links instead of relying on familiar repository names.

## Decisions and outstanding validation

The resulting course has a 16-week spine with equal emphasis on interpreting traces and building bounded components. Required readings are deliberately smaller than the reference catalog; advanced papers/repositories support projects. Rubin is the recurring hardware case and AMD supplies the portability comparison. CPU-only analysis remains useful but is never represented as a GPU performance experiment.

Before executable release: pilot lab durations; choose accessible model weights and datasets with compatible licenses; pin separate lab environments; validate selected CUDA/ROCm architectures and numerical paths; record backend dispatch; obtain representative multi-GPU traces; and test checkpoint recovery and serving load generators. These are implementation gates, not claims of completed validation. See [ROADMAP.md](../ROADMAP.md).

## Shared-discussion follow-up — 2026-10-27

Read the relevant shared-chat sections on inference-service opportunities, automated optimization, AOTInductor/Triton, deployment backends and TileLang. Verified named projects against primary README/tool documentation, deployment docs, paper abstracts/scope and vLLM issue status. Added a [technique map](../notes/optimization_techniques.md) and [coverage audit](../notes/course_component_audit.md). Full paper reproductions, new engine/DSL installations and performance claims are not part of this follow-up. The private/shared conversation itself is not copied into the public repository; technical statements cite their primary sources.
