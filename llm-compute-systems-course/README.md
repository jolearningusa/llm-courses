# Efficient LLM Computing Systems: Inference, GPU Kernels & Training

A systems engineering course on building and measuring the computing stack for LLM inference, from PyTorch and compilers through CUDA kernels, memory, operating systems and serving runtimes to deployed programs, with distributed training at scale as the final part and NVIDIA Vera Rubin as a recurring case study.

**Status: materials and CPU-reference release, 2026-10-27.** The package contains 36 decks with 36 editable SVG teaching illustrations, L0–L15 handouts/reference kits, a capstone and 20 original project specifications, expanded into [25 detailed project execution plans](projects/COURSE_APPENDIX.md). [Validation](VALIDATION.md) distinguishes executed CPU paths, analytical/source work and unverified accelerated extensions. GPU qualification and a student pacing pilot remain pending.

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
9. [Slide and lab build plan](BUILD_PLAN.md): 32 core decks plus four supplements (optimization decisions; programming models beyond CUDA; model lowering and co-design; inference engines compared), L0–L15 kits, serial PR order and release gates.

10. [Release guide](COURSE_RELEASE.md), [validation record](VALIDATION.md), [SVG illustration library](slides/illustrations/) and [capstone](labs/kit/capstone/CAPSTONE.md).

11. [Appendix A — project ideas and execution plans](projects/COURSE_APPENDIX.md): 25 engineering/research options, weekly plans, baselines, hardware gates and deliverables; [paper review](references/project_research_review.md).

## Audience and prerequisites

Systems engineers and graduate students comfortable with Python, basic C/C++, linear algebra, processes/threads, virtual memory and elementary distributed systems. Prior CUDA experience is helpful but not required. Students should read tensor shapes and a forward/backward pass; designing architectures, selecting pretraining data and optimizing model quality are outside the central syllabus.

The brief architecture primer is intentional: understand *why* MoE creates all-to-all traffic or GQA reduces cache bytes, then spend the course implementing and measuring the execution stack. The existing [AgentOS course](../llm-agentos-course/) covers the agent harness above this stack; this course covers the computation beneath it.

## Course map

The course is inference-first. Parts I–V follow one request's computation from the workload and a single GPU to a serving engine and a deployed program; Part VI, training at scale, comes last and reuses the same kernels, precision formats and measurement discipline. Deck and lab numbers follow the teaching order.

| Part | Weeks | Decks | Labs | Engineering question |
|---|---|---|---|---|
| I. Foundations: workloads and measurement | 1–2 | 01, 02, 03, 04 | L0–L1 | What work, data movement and memory capacity does this job require, and how do we measure it honestly? |
| II. Single-GPU execution: PyTorch, compilers and kernels | 3–6 | 05, 06, 07, 33, 08, 09, 10, 11, 12, 32 | L2–L5 | How does a tensor program become efficient work on one device? |
| III. Numerics and quantization | 7 | 13, 14 | L6 | Which formats, scales and accumulations keep a model correct while moving fewer bytes? |
| IV. Inference serving | 8–11 | 15, 16, 17, 18, 19, 20, 21 | L7–L10 | How do we deliver useful tokens under latency, memory and load constraints? |
| V. Deployment and platforms | 11–12 | 22, 23, 24 | L11, L15A | How do we package a program for a target, and what does the target platform provide? |
| VI. Training at scale | 13–15 | 25, 26, 27, 28, 29, 30 | L12–L14, L6B, L15B | How do we fit, synchronize and recover a training job, and does a component gain improve the whole job? |
| Capstone | 16 | 31 | Capstone | Which change survives a credible baseline, correctness checks and an adverse case? |

Supplements sit next to the material they extend: deck 33 (programming models beyond CUDA) follows deck 07, and deck 32 (optimization decisions and automated search) follows Part II and is revisited after Part V. The order follows the same spine as CMU 15-442, Stanford CS336's systems unit and *How To Scale Your Model* (single-device execution before scale-out), with inference first as in Stanford CS349E (inference only) and MIT 6.5940 (efficient inference before efficient training). The comparison and sources are in the [research review](references/research_review.md#course-structure-compared-with-other-courses).

Each lab follows **predict → implement → validate correctness → profile → explain → compare end-to-end**. A speedup without a stated baseline, error check and workload is incomplete.

## Learning outcomes

- Derive memory and communication budgets for training, prefill and decode, and test them against traces.
- Write and tune CUDA reductions and tiled matrix operations; explain occupancy, bandwidth, synchronization and numerical constraints.
- Trace an operation through PyTorch, compiler IR, library/custom kernel dispatch and CUDA execution.
- Compare eager, compiled and ahead-of-time deployment, including the costs of compilation and shape specialization.
- Explain and measure KV paging, prefix reuse, chunked prefill, speculative decoding and disaggregated serving.
- Diagnose CPU, NUMA, input, memory-allocation, communication and checkpoint bottlenecks alongside GPU kernels.
- Distinguish semantics-preserving execution optimizations from numerical approximation or architecture changes.
- Build distributed training experiments with DDP/FSDP2, tensor/pipeline/context/expert parallelism, activation checkpointing and recovery.

## Hardware access

| Tier | What students can do | Limits |
|---|---|---|
| CPU laptop | Budget calculations, trace analysis, eager/export experiments and small ExecuTorch CPU workloads where supported | Does not validate CUDA or distributed performance |
| One CUDA GPU | Kernel, compiler, attention, precision and small-model serving labs | Tensor Core/TMA/low-precision paths depend on GPU and software support |
| 2–8 GPUs | DDP/FSDP2 and parallelism, collectives and MoE experiments | Multi-node networking needs a suitable interconnect |
| Shared cluster / Rubin access | Topology-aware integration and architecture-specific validation | Optional; analytical projections remain explicitly labeled estimates |

The course does not depend on a Rubin GPU. Use the available device for measurements and Rubin for source-grounded analysis. Release each executable lab with its own tested environment; do not assume all projects share compatible latest versions.

## Slide decks

Decks are listed in teaching order, grouped by part. Each row links the English deck and, on that same row, the Simplified Chinese deck (`slides/zh/`, same file name).

| Deck | When |
|---|---|
| [Efficient LLM Computing Systems](slides/00_master_compute_systems.html) (15 slides) · [简体中文](slides/zh/00_master_compute_systems.html) | Course map and deck directory |
| **Part I · Foundations: workloads and measurement (weeks 1–2)** | |
| [Workloads, Rooflines and Capacity](slides/01_workloads_rooflines.html) (17 slides) · [简体中文](slides/zh/01_workloads_rooflines.html) | Week 1, lecture A |
| [Architecture as a Computing Workload](slides/02_architecture_workload_primer.html) (13 slides) · [简体中文](slides/zh/02_architecture_workload_primer.html) | Week 1, lecture B |
| [Measurement and Profiling](slides/03_measurement_profiling.html) (17 slides) · [简体中文](slides/zh/03_measurement_profiling.html) | Week 2, lecture A |
| [Host, OS and Data Paths](slides/04_host_os_data_paths.html) (17 slides) · [简体中文](slides/zh/04_host_os_data_paths.html) | Week 2, lecture B |
| **Part II · Single-GPU execution: PyTorch, compilers and kernels (weeks 3–6)** | |
| [PyTorch Execution and Autograd](slides/05_pytorch_execution_autograd.html) (18 slides) · [简体中文](slides/zh/05_pytorch_execution_autograd.html) | Week 3, lecture A |
| [PyTorch Compiler and Specialization](slides/06_pytorch_compiler.html) (18 slides) · [简体中文](slides/zh/06_pytorch_compiler.html) | Week 3, lecture B |
| [CUDA Execution and Memory](slides/07_cuda_execution_memory.html) (17 slides) · [简体中文](slides/zh/07_cuda_execution_memory.html) | Week 4, lecture A |
| [Programming Models beyond CUDA](slides/33_programming_models_beyond_cuda.html) (19 slides) · [简体中文](slides/zh/33_programming_models_beyond_cuda.html) | Supplement to Part II, after deck 07 |
| [Reductions and Custom Operator Contracts](slides/08_reductions_custom_operators.html) (17 slides) · [简体中文](slides/zh/08_reductions_custom_operators.html) | Week 4, lecture B |
| [GEMM Tiling and Tensor Cores](slides/09_gemm_tensor_cores.html) (19 slides) · [简体中文](slides/zh/09_gemm_tensor_cores.html) | Week 5, lecture A |
| [Asynchronous Kernel Pipelines](slides/10_async_kernel_pipelines.html) (20 slides) · [简体中文](slides/zh/10_async_kernel_pipelines.html) | Week 5, lecture B |
| [Attention Execution and Online Softmax](slides/11_attention_execution.html) (17 slides) · [简体中文](slides/zh/11_attention_execution.html) | Week 6, lecture A |
| [Attention Kernel and Hardware Co-design](slides/12_attention_kernel_codesign.html) (17 slides) · [简体中文](slides/zh/12_attention_kernel_codesign.html) | Week 6, lecture B |
| [Optimization Decisions and Automated Performance Engineering](slides/32_optimization_decision_systems.html) (23 slides) · [简体中文](slides/zh/32_optimization_decision_systems.html) | Supplement after Part II; revisit after Part V |
| **Part III · Numerics and quantization (week 7)** | |
| [Precision, Scales and Numerical Contracts](slides/13_precision_numerics.html) (17 slides) · [简体中文](slides/zh/13_precision_numerics.html) | Week 7, lecture A |
| [Quantization and Low-Precision Serving](slides/14_quantization_serving.html) (15 slides) · [简体中文](slides/zh/14_quantization_serving.html) | Week 7, lecture B |
| **Part IV · Inference serving (weeks 8–11)** | |
| [Serving Iterations, Admission and State Ownership](slides/15_serving_scheduler.html) (17 slides) · [简体中文](slides/zh/15_serving_scheduler.html) | Week 8, lecture A |
| [KV, Prefix and Response Caches across Inference Stacks](slides/16_kv_prefix_cache_mechanisms.html) (17 slides) · [简体中文](slides/zh/16_kv_prefix_cache_mechanisms.html) | Week 8, lecture B |
| [Inference Engines Compared: llama.cpp, vLLM and the Rest](slides/35_inference_engines_compared.html) (19 slides) · [简体中文](slides/zh/35_inference_engines_compared.html) | Supplement to Part IV, after deck 16 |
| [Prefill, Decode and Latency-Constrained Scheduling](slides/17_prefill_decode_scheduling.html) (17 slides) · [简体中文](slides/zh/17_prefill_decode_scheduling.html) | Week 9, lecture A |
| [Disaggregated Inference and KV Handoff](slides/18_disaggregated_inference.html) (17 slides) · [简体中文](slides/zh/18_disaggregated_inference.html) | Week 9, lecture B |
| [MoE Routing, Dispatch and Grouped Compute](slides/19_moe_execution.html) (17 slides) · [简体中文](slides/zh/19_moe_execution.html) | Week 10, lecture A |
| [Speculative Decoding and Correct Verification](slides/20_speculative_decoding.html) (17 slides) · [简体中文](slides/zh/20_speculative_decoding.html) | Week 10, lecture B |
| [AgentOS Workloads and Cache-Lifecycle Policies](slides/21_agentos_inference_cache_policy.html) (17 slides) · [简体中文](slides/zh/21_agentos_inference_cache_policy.html) | Week 11, lecture A |
| **Part V · Deployment and platforms (weeks 11–12)** | |
| [Export Contracts and AOTInductor Deployment](slides/22_export_aotinductor.html) (17 slides) · [简体中文](slides/zh/22_export_aotinductor.html) | Week 11, lecture B |
| [ExecuTorch Runtime, Delegation and Memory Planning](slides/23_executorch_deployment.html) (18 slides) · [简体中文](slides/zh/23_executorch_deployment.html) | Week 12, lecture A |
| [Model Lowering and Hardware Co-design](slides/34_model_lowering_codesign.html) (13 slides) · [简体中文](slides/zh/34_model_lowering_codesign.html) | Supplement to Part V, after deck 23 |
| [Vera Rubin and NVIDIA/AMD Software Stacks](slides/24_rubin_vendor_stacks.html) (17 slides) · [简体中文](slides/zh/24_rubin_vendor_stacks.html) | Week 12, lecture B |
| **Part VI · Training at scale (weeks 13–15)** | |
| [Training Precision, DDP, ZeRO and Fully Sharded Training](slides/25_ddp_zero_fsdp.html) (22 slides) · [简体中文](slides/zh/25_ddp_zero_fsdp.html) | Week 13, lecture A |
| [Collectives, Topology and Exposed Communication](slides/26_collectives_overlap.html) (17 slides) · [简体中文](slides/zh/26_collectives_overlap.html) | Week 13, lecture B |
| [Parallel Layouts and Recomputation](slides/27_parallelism_recomputation.html) (17 slides) · [简体中文](slides/zh/27_parallelism_recomputation.html) | Week 14, lecture A |
| [Reading a Production Training Stack](slides/28_training_stack_walkthrough.html) (17 slides) · [简体中文](slides/zh/28_training_stack_walkthrough.html) | Week 14, lecture B |
| [Checkpointing, Input State and Useful Training Work](slides/29_checkpoint_recovery.html) (17 slides) · [简体中文](slides/zh/29_checkpoint_recovery.html) | Week 15, lecture A |
| [Whole-System Efficiency and Useful Work](slides/30_whole_system_efficiency.html) (17 slides) · [简体中文](slides/zh/30_whole_system_efficiency.html) | Week 15, lecture B |
| **Capstone (week 16)** | |
| [Capstone Workshop: Evidence to Decision](slides/31_capstone_workshop.html) (11 slides) · [简体中文](slides/zh/31_capstone_workshop.html) | Week 16 |

Planned topics and release dependencies: [build plan](BUILD_PLAN.md).

## Assessment

Suggested weighting: labs 45%, paper/trace critiques 15%, capstone 40%. Correctness is a gate for performance credit. Grade causal explanation, reproducibility and the training/inference objective, rather than requiring a fixed speedup on heterogeneous hardware. The [syllabus](SYLLABUS.md) defines the capstone and workload policy.

## Repository layout

```text
llm-compute-systems-course/
├── labs/
│   ├── kit/
│   │   ├── capstone/
│   │   │   ├── CAPSTONE.md
│   │   │   ├── capstone_manifest.example.json
│   │   │   └── critique_example.md
│   │   ├── cuda/
│   │   │   └── rmsnorm.cu
│   │   ├── examples/
│   │   │   ├── l11_aoti_cpu.json
│   │   │   └── l11_executorch_cpu.json
│   │   ├── handouts/
│   │   │   ├── lab00_budgets.md
│   │   │   ├── lab01_measurement.md
│   │   │   ├── lab02_pytorch.md
│   │   │   ├── lab03_reductions.md
│   │   │   ├── lab04_gemm.md
│   │   │   ├── lab05_attention.md
│   │   │   ├── lab06_precision.md
│   │   │   ├── lab07_cache.md
│   │   │   ├── lab08_scheduling.md
│   │   │   ├── lab09_moe.md
│   │   │   ├── lab10_agent_policy.md
│   │   │   ├── lab11_deployment.md
│   │   │   ├── lab12_distributed.md
│   │   │   ├── lab13_parallelism.md
│   │   │   ├── lab14_recovery.md
│   │   │   └── lab15_vendor_efficiency.md
│   │   ├── tests/
│   │   │   ├── test_agent_policy.py
│   │   │   ├── test_attention.py
│   │   │   ├── test_budgets.py
│   │   │   ├── test_cache.py
│   │   │   ├── test_deployment.py
│   │   │   ├── test_distributed.py
│   │   │   ├── test_efficiency.py
│   │   │   ├── test_gemm.py
│   │   │   ├── test_measurement.py
│   │   │   ├── test_moe_recovery.py
│   │   │   ├── test_parallelism.py
│   │   │   ├── test_precision.py
│   │   │   ├── test_pytorch_lab.py
│   │   │   ├── test_reductions.py
│   │   │   └── test_scheduling.py
│   │   ├── README.md
│   │   ├── agent_policy_lab.py
│   │   ├── attention_lab.py
│   │   ├── budgets.py
│   │   ├── cache_lab.py
│   │   ├── deployment_lab.py
│   │   ├── distributed_lab.py
│   │   ├── gemm_lab.py
│   │   ├── measurement.py
│   │   ├── moe_lab.py
│   │   ├── parallelism_lab.py
│   │   ├── precision_lab.py
│   │   ├── pytorch_lab.py
│   │   ├── recovery_lab.py
│   │   ├── reductions.py
│   │   ├── requirements-executorch-lock.txt
│   │   ├── requirements-executorch.txt
│   │   ├── requirements-pytorch-lock.txt
│   │   ├── requirements-pytorch.txt
│   │   ├── scheduling_lab.py
│   │   ├── system_efficiency.py
│   │   ├── training_source_walkthrough.md
│   │   └── vendor_dossier_template.md
│   ├── LAB_PLAN.md
│   └── report_template.md
├── notes/
│   ├── agentic_caching.md
│   ├── amd_stack.md
│   ├── project_appendix_review.md
│   ├── course_component_audit.md
│   ├── nvidia_stack.md
│   ├── optimization_techniques.md
│   ├── pytorch_stack.md
│   ├── vera_rubin.md
│   └── workload_primer.md
├── projects/
│   ├── COURSE_APPENDIX.md
│   ├── POTENTIAL_PROJECTS.md
│   └── optimization-engine.svg
├── references/
│   ├── project_research_review.md
│   ├── bibliography_and_links.md
│   ├── research_review.md
│   └── tracked_sources.json
├── slides/
│   ├── illustrations/
│   │   ├── README.md
│   │   ├── agent-wait.svg
│   │   ├── amdahl-ledger.svg
│   │   ├── aoti-artifact.svg
│   │   ├── attention-pipeline.svg
│   │   ├── autograd-graph.svg
│   │   ├── build_illustrations.py
│   │   ├── capstone-evidence.svg
│   │   ├── chunked-prefill.svg
│   │   ├── collective-ring.svg
│   │   ├── compiler-path.svg
│   │   ├── cuda-hierarchy.svg
│   │   ├── double-buffer.svg
│   │   ├── engine-design-points.svg
│   │   ├── execution-stack.svg
│   │   ├── executorch-partitions.svg
│   │   ├── expert-skew.svg
│   │   ├── fsdp-shards.svg
│   │   ├── gemm-tiles.svg
│   │   ├── host-data-path.svg
│   │   ├── lowering-pipeline.svg
│   │   ├── manifest.json
│   │   ├── moe-primer.svg
│   │   ├── online-attention.svg
│   │   ├── optimization-choices.svg
│   │   ├── parallel-mesh.svg
│   │   ├── pd-placement.svg
│   │   ├── precision-lifecycle.svg
│   │   ├── prefix-branches.svg
│   │   ├── programming-model-layers.svg
│   │   ├── quantized-bytes.svg
│   │   ├── reduction-tree.svg
│   │   ├── restart-state.svg
│   │   ├── roofline.svg
│   │   ├── rubin-data-paths.svg
│   │   ├── shared-blocks.svg
│   │   ├── source-walk.svg
│   │   ├── speculative-correction.svg
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
│   ├── 14_quantization_serving.html
│   ├── 15_serving_scheduler.html
│   ├── 16_kv_prefix_cache_mechanisms.html
│   ├── 17_prefill_decode_scheduling.html
│   ├── 18_disaggregated_inference.html
│   ├── 19_moe_execution.html
│   ├── 20_speculative_decoding.html
│   ├── 21_agentos_inference_cache_policy.html
│   ├── 22_export_aotinductor.html
│   ├── 23_executorch_deployment.html
│   ├── 24_rubin_vendor_stacks.html
│   ├── 25_ddp_zero_fsdp.html
│   ├── 26_collectives_overlap.html
│   ├── 27_parallelism_recomputation.html
│   ├── 28_training_stack_walkthrough.html
│   ├── 29_checkpoint_recovery.html
│   ├── 30_whole_system_efficiency.html
│   ├── 31_capstone_workshop.html
│   ├── 32_optimization_decision_systems.html
│   ├── 33_programming_models_beyond_cuda.html
│   ├── 34_model_lowering_codesign.html
│   ├── 35_inference_engines_compared.html
│   └── zh/                                    # Simplified Chinese (zh-CN) twin of each deck above; same file name
│       ├── 00_master_compute_systems.html
│       ├── 01_workloads_rooflines.html
│       ├── 02_architecture_workload_primer.html
│       ├── 03_measurement_profiling.html
│       ├── 04_host_os_data_paths.html
│       ├── 05_pytorch_execution_autograd.html
│       ├── 06_pytorch_compiler.html
│       ├── 07_cuda_execution_memory.html
│       ├── 08_reductions_custom_operators.html
│       ├── 09_gemm_tensor_cores.html
│       ├── 10_async_kernel_pipelines.html
│       ├── 11_attention_execution.html
│       ├── 12_attention_kernel_codesign.html
│       ├── 13_precision_numerics.html
│       ├── 14_quantization_serving.html
│       ├── 15_serving_scheduler.html
│       ├── 16_kv_prefix_cache_mechanisms.html
│       ├── 17_prefill_decode_scheduling.html
│       ├── 18_disaggregated_inference.html
│       ├── 19_moe_execution.html
│       ├── 20_speculative_decoding.html
│       ├── 21_agentos_inference_cache_policy.html
│       ├── 22_export_aotinductor.html
│       ├── 23_executorch_deployment.html
│       ├── 24_rubin_vendor_stacks.html
│       ├── 25_ddp_zero_fsdp.html
│       ├── 26_collectives_overlap.html
│       ├── 27_parallelism_recomputation.html
│       ├── 28_training_stack_walkthrough.html
│       ├── 29_checkpoint_recovery.html
│       ├── 30_whole_system_efficiency.html
│       ├── 31_capstone_workshop.html
│       ├── 32_optimization_decision_systems.html
│       ├── 33_programming_models_beyond_cuda.html
│       ├── 34_model_lowering_codesign.html
│       └── 35_inference_engines_compared.html
├── BUILD_PLAN.md
├── COURSE_RELEASE.md
├── LICENSE
├── README.md
├── ROADMAP.md
├── SYLLABUS.md
└── VALIDATION.md
```

## Source and version policy

Readings were checked against primary web sources on 2026-10-26. Review depth and limitations are stated in the research review. Software documentation and repository main branches are evidence of documented behavior, not a tested course environment. No package-version compatibility matrix or hardware benchmark is asserted by this blueprint. Add `references/tracked_sources.json` when lab APIs are pinned and executed, following [the repository update process](../docs/updating-courses.md).
