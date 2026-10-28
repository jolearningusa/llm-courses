# Efficient LLM Computing Systems: Training, Inference & GPU Kernels

A systems engineering course on building and measuring the computing stack for LLM training and inference, from PyTorch and compilers through CUDA kernels, memory, operating systems, interconnects and cluster runtimes, using NVIDIA Vera Rubin as a recurring case study.

**Status: incremental teaching releases in progress, 2026-10-26.** The deck table and lab-kit index identify supplied materials. The build plan records serial PR progress and distinguishes validated CPU paths, analytical exercises, student extensions and hardware validation still needed. No GPU benchmark is claimed without an explicit release record.

## Start here

1. [Research review and design decisions](references/research_review.md): what existing courses, papers and repositories contribute, where they fall short for this course, and what to reproduce.
2. [Annotated bibliography](references/bibliography_and_links.md): primary papers, official tutorials and upstream repositories, with required versus optional readings.
3. [Detailed syllabus](SYLLABUS.md): 16 weeks, two lectures per week, assessed outcomes and a common training/inference experiment.
4. [Workload and algorithm primer](notes/workload_primer.md): the minimum dense/MoE and attention background needed to understand the computation.
5. [PyTorch stack and deployment artifacts](notes/pytorch_stack.md): eager/autograd, compilation, export, AOTInductor, TorchAO and ExecuTorch.
6. [Vera Rubin case study](notes/vera_rubin.md): documented mechanisms, an assumptions ledger and exercises that do not require owning a rack.
   Compare the [NVIDIA library map](notes/nvidia_stack.md) with the [AMD ROCm stack](notes/amd_stack.md).
7. [AgentOS inference and cache mechanisms](notes/agentic_caching.md): vLLM, llama.cpp and SGLang, tool waits and cache policy.
8. [Lab plan](labs/LAB_PLAN.md) and [experiment report](labs/report_template.md).
9. [Slide and lab build plan](BUILD_PLAN.md): 32 planned decks, L0–L15 kits, serial PR order and release gates.

## Audience and prerequisites

Systems engineers and graduate students comfortable with Python, basic C/C++, linear algebra, processes/threads, virtual memory and elementary distributed systems. Prior CUDA experience is helpful but not required. Students should read tensor shapes and a forward/backward pass; designing architectures, selecting pretraining data and optimizing model quality are outside the central syllabus.

The brief architecture primer is intentional: understand *why* MoE creates all-to-all traffic or GQA reduces cache bytes, then spend the course implementing and measuring the execution stack. The existing [AgentOS course](../llm-agentos-course/) covers the agent harness above this stack; this course covers the computation beneath it.

## Course map

| Block | Weeks | Engineering question |
|---|---|---|
| Workloads and measurement | 1–2 | What work, data movement and memory capacity does this job require? |
| Frameworks, compilers and kernels | 3–7 | How does a tensor program become efficient executable work? |
| Distributed training | 8–10 | How do we fit, synchronize and recover a training job? |
| Serving and deployment | 11–14 | How do we deliver useful tokens under latency, memory and power constraints? |
| Whole-system integration | 15–16 | Which cross-layer change improves the actual training or inference objective? |

Each lab follows **predict → implement → validate correctness → profile → explain → compare end-to-end**. A speedup without a stated baseline, error check and workload is incomplete.

## Learning outcomes

- Derive memory and communication budgets for training, prefill and decode, and test them against traces.
- Write and tune CUDA reductions and tiled matrix operations; explain occupancy, bandwidth, synchronization and numerical constraints.
- Trace an operation through PyTorch, compiler IR, library/custom kernel dispatch and CUDA execution.
- Compare eager, compiled and ahead-of-time deployment, including the costs of compilation and shape specialization.
- Build distributed training experiments with DDP/FSDP2, tensor/pipeline/context/expert parallelism, activation checkpointing and recovery.
- Explain and measure KV paging, prefix reuse, chunked prefill, speculative decoding and disaggregated serving.
- Diagnose CPU, NUMA, input, memory-allocation, communication and checkpoint bottlenecks alongside GPU kernels.
- Distinguish semantics-preserving execution optimizations from numerical approximation or architecture changes.

## Hardware access

| Tier | What students can do | Limits |
|---|---|---|
| CPU laptop | Budget calculations, trace analysis, eager/export experiments and small ExecuTorch CPU workloads where supported | Does not validate CUDA or distributed performance |
| One CUDA GPU | Kernel, compiler, attention, precision and small-model serving labs | Tensor Core/TMA/low-precision paths depend on GPU and software support |
| 2–8 GPUs | DDP/FSDP2 and parallelism, collectives and MoE experiments | Multi-node networking needs a suitable interconnect |
| Shared cluster / Rubin access | Topology-aware integration and architecture-specific validation | Optional; analytical projections remain explicitly labeled estimates |

The course does not depend on a Rubin GPU. Use the available device for measurements and Rubin for source-grounded analysis. Release each executable lab with its own tested environment; do not assume all projects share compatible latest versions.

## Slide decks

| Deck | Scope |
|---|---|
| [Efficient LLM Computing Systems](slides/00_master_compute_systems.html) (14 slides) | Course map and released-deck navigation |
| [Workloads, Rooflines and Capacity](slides/01_workloads_rooflines.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Architecture as a Computing Workload](slides/02_architecture_workload_primer.html) (13 slides) | Topic lecture, worked examples and practical exercises |
| [Measurement and Profiling](slides/03_measurement_profiling.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Host, OS and Data Paths](slides/04_host_os_data_paths.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [PyTorch Execution and Autograd](slides/05_pytorch_execution_autograd.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [PyTorch Compiler and Specialization](slides/06_pytorch_compiler.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [CUDA Execution and Memory](slides/07_cuda_execution_memory.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Reductions and Custom Operator Contracts](slides/08_reductions_custom_operators.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [GEMM Tiling and Tensor Cores](slides/09_gemm_tensor_cores.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Asynchronous Kernel Pipelines](slides/10_async_kernel_pipelines.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Attention Execution and Online Softmax](slides/11_attention_execution.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Attention Kernel and Hardware Co-design](slides/12_attention_kernel_codesign.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Precision, Scales and Numerical Contracts](slides/13_precision_numerics.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Precision in Training and Serving](slides/14_precision_training_serving.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [DDP, ZeRO and Fully Sharded Training](slides/15_ddp_zero_fsdp.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Collectives, Topology and Exposed Communication](slides/16_collectives_overlap.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Parallel Layouts and Recomputation](slides/17_parallelism_recomputation.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Reading a Production Training Stack](slides/18_training_stack_walkthrough.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [MoE Routing, Dispatch and Grouped Compute](slides/19_moe_execution.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Checkpointing, Input State and Useful Training Work](slides/20_checkpoint_recovery.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Serving Iterations, Admission and State Ownership](slides/21_serving_scheduler.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [KV, Prefix and Response Caches across Inference Stacks](slides/22_kv_prefix_cache_mechanisms.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Prefill, Decode and Latency-Constrained Scheduling](slides/23_prefill_decode_scheduling.html) (17 slides) | Topic lecture, worked examples and practical exercises |
| [Disaggregated Inference and KV Handoff](slides/24_disaggregated_inference.html) (17 slides) | Topic lecture, worked examples and practical exercises |

Planned topics and release dependencies: [build plan](BUILD_PLAN.md).

## Assessment

Suggested weighting: labs 45%, paper/trace critiques 15%, capstone 40%. Correctness is a gate for performance credit. Grade causal explanation, reproducibility and the training/inference objective, rather than requiring a fixed speedup on heterogeneous hardware. The [syllabus](SYLLABUS.md) defines the capstone and workload policy.

## Repository layout

```text
llm-compute-systems-course/
├── labs/
│   ├── kit/
│   │   ├── cuda/
│   │   │   └── rmsnorm.cu
│   │   ├── handouts/
│   │   │   ├── lab00_budgets.md
│   │   │   ├── lab01_measurement.md
│   │   │   ├── lab02_pytorch.md
│   │   │   ├── lab03_reductions.md
│   │   │   ├── lab04_gemm.md
│   │   │   ├── lab05_attention.md
│   │   │   ├── lab06_precision.md
│   │   │   ├── lab07_distributed.md
│   │   │   ├── lab08_parallelism.md
│   │   │   ├── lab09_moe.md
│   │   │   ├── lab10_recovery.md
│   │   │   ├── lab11_cache.md
│   │   │   └── lab12_scheduling.md
│   │   ├── tests/
│   │   │   ├── test_attention.py
│   │   │   ├── test_budgets.py
│   │   │   ├── test_cache.py
│   │   │   ├── test_distributed.py
│   │   │   ├── test_gemm.py
│   │   │   ├── test_measurement.py
│   │   │   ├── test_moe_recovery.py
│   │   │   ├── test_parallelism.py
│   │   │   ├── test_precision.py
│   │   │   ├── test_pytorch_lab.py
│   │   │   ├── test_reductions.py
│   │   │   └── test_scheduling.py
│   │   ├── README.md
│   │   ├── attention_lab.py
│   │   ├── budgets.py
│   │   ├── cache_lab.py
│   │   ├── distributed_lab.py
│   │   ├── gemm_lab.py
│   │   ├── measurement.py
│   │   ├── moe_lab.py
│   │   ├── parallelism_lab.py
│   │   ├── precision_lab.py
│   │   ├── pytorch_lab.py
│   │   ├── recovery_lab.py
│   │   ├── reductions.py
│   │   ├── requirements-pytorch-lock.txt
│   │   ├── requirements-pytorch.txt
│   │   ├── scheduling_lab.py
│   │   └── training_source_walkthrough.md
│   ├── LAB_PLAN.md
│   └── report_template.md
├── notes/
│   ├── agentic_caching.md
│   ├── amd_stack.md
│   ├── nvidia_stack.md
│   ├── pytorch_stack.md
│   ├── vera_rubin.md
│   └── workload_primer.md
├── references/
│   ├── bibliography_and_links.md
│   ├── research_review.md
│   └── tracked_sources.json
├── slides/
│   ├── illustrations/
│   │   ├── README.md
│   │   ├── attention-pipeline.svg
│   │   ├── autograd-graph.svg
│   │   ├── build_illustrations.py
│   │   ├── chunked-prefill.svg
│   │   ├── collective-ring.svg
│   │   ├── compiler-path.svg
│   │   ├── cuda-hierarchy.svg
│   │   ├── double-buffer.svg
│   │   ├── execution-stack.svg
│   │   ├── expert-skew.svg
│   │   ├── fsdp-shards.svg
│   │   ├── gemm-tiles.svg
│   │   ├── host-data-path.svg
│   │   ├── manifest.json
│   │   ├── moe-primer.svg
│   │   ├── online-attention.svg
│   │   ├── parallel-mesh.svg
│   │   ├── pd-placement.svg
│   │   ├── precision-lifecycle.svg
│   │   ├── prefix-branches.svg
│   │   ├── quantized-bytes.svg
│   │   ├── reduction-tree.svg
│   │   ├── restart-state.svg
│   │   ├── roofline.svg
│   │   ├── shared-blocks.svg
│   │   ├── source-walk.svg
│   │   └── timing-boundaries.svg
│   ├── 00_master_compute_systems.html
│   ├── 01_workloads_rooflines.html
│   ├── 02_architecture_workload_primer.html
│   ├── 03_measurement_profiling.html
│   ├── 04_host_os_data_paths.html
│   ├── 05_pytorch_execution_autograd.html
│   ├── 06_pytorch_compiler.html
│   ├── 07_cuda_execution_memory.html
│   ├── 08_reductions_custom_operators.html
│   ├── 09_gemm_tensor_cores.html
│   ├── 10_async_kernel_pipelines.html
│   ├── 11_attention_execution.html
│   ├── 12_attention_kernel_codesign.html
│   ├── 13_precision_numerics.html
│   ├── 14_precision_training_serving.html
│   ├── 15_ddp_zero_fsdp.html
│   ├── 16_collectives_overlap.html
│   ├── 17_parallelism_recomputation.html
│   ├── 18_training_stack_walkthrough.html
│   ├── 19_moe_execution.html
│   ├── 20_checkpoint_recovery.html
│   ├── 21_serving_scheduler.html
│   ├── 22_kv_prefix_cache_mechanisms.html
│   ├── 23_prefill_decode_scheduling.html
│   └── 24_disaggregated_inference.html
├── BUILD_PLAN.md
├── LICENSE
├── README.md
├── ROADMAP.md
└── SYLLABUS.md
```

## Source and version policy

Readings were checked against primary web sources on 2026-10-26. Review depth and limitations are stated in the research review. Software documentation and repository main branches are evidence of documented behavior, not a tested course environment. No package-version compatibility matrix or hardware benchmark is asserted by this blueprint. Add `references/tracked_sources.json` when lab APIs are pinned and executed, following [the repository update process](../docs/updating-courses.md).
