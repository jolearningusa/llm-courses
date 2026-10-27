# Slide deck and lab build plan

This is the implementation checklist for the [syllabus](SYLLABUS.md), with one deck per lecture topic and a paired practical assignment. The [lab specifications](labs/LAB_PLAN.md) define the experiments; this plan defines what to author, in what order, and how to decide that each unit is complete.

**Delivery workflow:** implement one bounded unit → check locally → open one PR → wait for required checks → squash merge → fetch and fast-forward `main` → create the next branch from latest `origin/main`. The user's 2026-10-26 instruction authorizes this sequential progression and merging green PRs. Do not stack unmerged branches. The repository's base branch is `main`, not `master`. Split an oversized unit into smaller serial PRs; never let an unmerged unit become the base for the next.

## Deliverables and status

- **32 planned decks:** a master overview, 30 topic lectures for weeks 1–15, and a capstone workshop. Week 16 uses the workshop and student presentations.
- **16 planned lab kits:** L0–L15, each with a handout, starter/reference material, runnable experiment or explicitly labeled analytical/trace path, validation and reporting instructions.
- **Shared infrastructure:** workload fixtures, environment manifests, metrics format, benchmark utilities, correctness helpers and release documentation.
- **Status at plan creation:** technical primers and lab specifications exist; no decks or executable kits have yet been released. Proposed filenames below are targets, not links to existing files. Update the progress table as units land.

## Deck contract

Use a self-contained HTML file in `slides/`, matching the repository's browser/keyboard conventions. Each deck starts at slide 1 and carries its own consistent total; the master links every released deck. Aim for 16–22 slides per topic, with 10–14 for the workload primer, 12–16 for the overview and 8–12 for the capstone workshop. These are authoring budgets, not promised counts; publish actual counts in the README only after implementation.

Each topic deck must include:

1. A concrete performance problem, prerequisites and two or three measurable learning outcomes.
2. A dataflow/timeline/memory diagram and one worked example with units and explicit assumptions.
3. The mechanism, its implementation location in the stack and relevant code/API excerpts at a recorded version.
4. A comparison with a baseline, one adverse case and a discussion of correctness/numerical constraints.
5. A trace-reading or calculation exercise, followed by the paired lab's inputs, measurements and acceptance criteria.
6. Primary references close to claims, with an annotated reading list entry; clearly labeled vendor claims, analytical estimates and measured results.

Keep the architecture primer short. Put CUDA/ROCm implementation details in the kernel lectures, and reference the same dense, decode and MoE workload shapes throughout. Explain vendor mappings by responsibility rather than claiming equivalent implementations.

## Topic-to-deck-to-practice manifest

All names below are planned files under `slides/`. A lab may support two related lectures through separate exercises; do not create 30 independent environments.

| Week | Planned deck | Required teaching artifact | Paired practice and evidence |
|---|---|---|---|
| Entry | `00_master_compute_systems.html` | Stack map, course paths, hardware tiers and links to released decks | Route students to setup and L0; distinguish released from planned material |
| 1A | `01_workloads_rooflines.html` | Forward/backward/update versus prefill/decode; worked FLOP/byte/capacity budgets | L0A: calculate bottlenecks and test sensitivity to achieved rates |
| 1B | `02_architecture_workload_primer.html` | Dense/MoE and attention/state comparison; exact versus changed semantics | L0B: MHA/GQA KV budget and resident/active MoE weight accounting |
| 2A | `03_measurement_profiling.html` | Asynchronous timing, warmup, compilation, peak memory and profiler perturbation | L1A: classify a timeline; compare host and device timing |
| 2B | `04_host_os_data_paths.html` | CPU/NUMA, page faults, pinning, DMA and GPU starvation diagram | L1B: diagnose an input/transfer/launch bottleneck; supply trace-only alternative |
| 3A | `05_pytorch_execution_autograd.html` | Storage/strides, dispatcher, saved tensors, backward and optimizer timeline | L2A: trace a block and inspect saved state/gradients |
| 3B | `06_pytorch_compiler.html` | Capture/guards/FX/Inductor and graph-break versus recompilation example | L2B: eager/compiled, fixed/dynamic shapes, cold/warm comparison |
| 4A | `07_cuda_execution_memory.html` | Warp/block mapping, coalescing, registers/shared memory and occupancy | L3A: simple correct reduction and declared input constraints |
| 4B | `08_reductions_custom_operators.html` | Stable reduction, synchronization, custom-op contract and backward | L3B: tune and validate tails/error/gradients; sanitizer instructions |
| 5A | `09_gemm_tensor_cores.html` | GEMM reuse/tiling/layout worked example and library baseline | L4A: training/prefill, decode and grouped shapes |
| 5B | `10_async_kernel_pipelines.html` | Double buffering, TMA, warp specialization and persistent scheduling | L4B: inspect generated/library execution; compare supported CUDA/Triton/CuTe/Tile paths |
| 6A | `11_attention_execution.html` | Online-softmax equations, tiled dataflow and backward/recompute costs | L5A: exact reference and tiled forward validation |
| 6B | `12_attention_kernel_codesign.html` | FA1–FA4 mechanism progression, softmax/matmul balance and ragged scheduling | L5B: matched-mask/shape backend comparison and memory/time scaling |
| 7A | `13_precision_numerics.html` | Range/scales/accumulation/packing and conversion-cost example | L6A: error and byte budgets before timing |
| 7B | `14_precision_training_serving.html` | AMP, TE, TorchAO, PTQ/QAT; training/serving support distinctions | L6B: one supported recipe with loss/quality and adverse-shape checks |
| 8A | `15_ddp_zero_fsdp.html` | Persistent/transient training state and all-gather/reduce-scatter timeline | L7A: DDP/FSDP2 at fixed effective batch |
| 8B | `16_collectives_overlap.html` | Collective payload/latency model, topology, buckets and exposed waits | L7B: overlap/placement ablation; NCCL/RCCL and device-initiated comparison |
| 9A | `17_parallelism_recomputation.html` | TP/PP/sequence/context layouts, microbatch bubbles and activation checkpoints | L8A: choose two feasible layouts under constraints |
| 9B | `18_training_stack_walkthrough.html` | Configuration → sharding → kernels → collectives in TorchTitan/Megatron | L8B: inspect recorded source/trace; explain each selected implementation |
| 10A | `19_moe_execution.html` | Routing/permutation/dispatch/GEMM/combine pipeline and skew | L9: uniform versus skewed experts; local and distributed paths |
| 10B | `20_checkpoint_recovery.html` | Input pipeline, persisted state, asynchronous save and recovery accounting | L10: interruption/resume and useful-throughput report |
| 11A | `21_serving_scheduler.html` | Request state machine, iteration batching, admission/cancellation/preemption | L11A: follow a request and release its resources correctly |
| 11B | `22_kv_prefix_cache_mechanisms.html` | vLLM blocks, SGLang radix paths, llama.cpp slots/RAM/persistence; response-cache distinction | L11B: cold/shared/mismatch/pressure/invalidation cache matrix |
| 12A | `23_prefill_decode_scheduling.html` | Interference, chunk budgets, queueing and SLO goodput | L12A: scheduling under fixed offered-load distributions |
| 12B | `24_disaggregated_inference.html` | KV handoff, routing/tiering, transfer-versus-recompute break-even | L12B: matched-resource colocated/split experiment or labeled simulation |
| 13A | `25_speculative_decoding.html` | Draft/verify/correct and acceptance/overhead calculation | L13A: correctness plus benefit across acceptance/load regimes |
| 13B | `26_agentos_inference_cache_policy.html` | Model/tool-wait/resume/branch trace, TTL and cache byte-seconds | L13B: retain/offload/recompute and task completion latency; invalidation across updates |
| 14A | `27_export_aotinductor.html` | Export contracts, compiled PT2 archives, producer/loader and cold start | L14A: eager/export/compile/AOT deployment comparison with out-of-contract shape |
| 14B | `28_executorch_deployment.html` | Target lowering/delegation, `.pte`/`.ptd`, memory planning and supported quantization | L14B: available CPU target first, then optional edge/GPU path |
| 15A | `29_rubin_vendor_stacks.html` | Rubin topology and hardware-to-NVIDIA-library map; AMD counterparts | L15A: two dataflows and software-support/portability dossier |
| 15B | `30_whole_system_efficiency.html` | Useful throughput, recovery/queueing/power boundaries and Amdahl example | L15B: reconcile component gains with job-level outcome; finalize L10 results |
| 16 | `31_capstone_workshop.html` | Proposal, ablations, reproducibility, rubric and example report critique | Training/serving/deployment capstone; final artifact and presentation |

## Lab kit build contract

Use a future shared `labs/kit/` for common fixtures, references and measurement utilities, and versioned handouts for L0–L15. Decide exact directory organization in the foundation PR and then document the actual files in the README tree. Do not add empty placeholders that suggest a lab runs.

Every released lab needs a student handout, explicit prerequisites, starter files or a bounded code-reading task, reference calculations/fixtures, correctness checks, a results schema, example report and reproducible run instructions. Include instructor guidance for common failures and a pilot-duration record. Label any completed reference implementation separately from student exercise tasks. Publish sample numbers only with their environment and provenance; never invent benchmark traces.

| Lab | Minimum build output | Hardware path | Release gate |
|---|---|---|---|
| L0 | Resource-budget calculator, worked fixtures, exercises and unit checks | CPU | Units, GQA example and dtype/sharding accounting validated |
| L1 | Timing/trace worksheet, timing harness and diagnostic experiment | CPU analysis; GPU measurement | Async timing and profiler overhead separated; any sample trace has provenance |
| L2 | Small PyTorch forward/training block, compiler experiment and logs | CPU; optional GPU | Outputs/gradients and shape-contract behavior checked; actual backend recorded |
| L3 | Reduction starter/reference, integration harness and correctness suite | CUDA | Tails/error constraints tested on declared hardware; sanitizer run or prerequisite explicitly noted |
| L4 | GEMM shape suite, library baseline and tuning exercises | CUDA; optional AMD | Compute/accumulation/layout and actual kernel dispatch documented |
| L5 | Exact attention reference, tiled exercise, backend benchmark | CPU reference; GPU performance | Masks/lengths/error checked; quadratic work distinguished from intermediate storage |
| L6 | Supported precision recipe, quality/loss fixtures and conversion timing | Device-specific | Numerical threshold, calibration/evaluation split and fallback behavior explicit |
| L7 | DDP/FSDP2 launch/configuration, memory/collective measurements | 2–8 GPUs; analytical alternative | Fixed effective batch, comparable training progression and transient peaks reconciled |
| L8 | Two feasible parallel plans and source/trace walkthrough | Multi-GPU; analytical alternative | Sharding, placement and bubble/communication costs explained |
| L9 | MoE routing fixture, grouped compute and phase instrumentation | One GPU; multi-GPU extension | Route/combine reference passes; skew and hot-expert costs visible |
| L10 | Checkpoint/resume harness and controlled interruption test | CPU example; multi-GPU validation | Optimizer/RNG/data position restored; lost/repeated work accounted |
| L11 | Cache workload matrix and two-engine comparison adapter/worksheet | llama.cpp CPU and/or GPU engines | Matched tokens, memory and invalidation checked; no cross-engine state-format assumption |
| L12 | Arrival replay, SLO metrics and scheduling/split-serving experiment | 2+ GPUs for measured split; simulation option | Same resource budget, offered load and failure/timeout accounting |
| L13 | Separate speculative and agent-wait exercises sharing report utilities | GPU; labeled simulation option | Correct verification/correction and cache-policy accounting; no model calls fabricated as measurements |
| L14 | Export/deploy experiment and artifact inspection | Supported CPU first; optional other targets | Producer/loader/contracts, cold/warm timings and numerical equivalence established |
| L15 | Rubin/NVIDIA/AMD dossier template and end-to-end efficiency worksheet | Analytical core; hardware extension | Primary facts cited, tested versus unverified support separated |

## Shared infrastructure to build first

The foundation should define deterministic shape/token/routing fixtures, a JSON result schema and environment manifest. GPU helpers must synchronize appropriately, distinguish allocation/warmup/build from timed work and report unsupported paths. Keep CPU reference functions dependency-light. A results record should identify lab/exercise, variant, workload, hardware, software, repetitions, correctness result, timing scope and measurement provenance.

Do not create one global dependency file that forces incompatible versions of PyTorch, every serving engine and all vendor libraries into one environment. Use a small common core with per-lab environment manifests. GPU availability must be detected explicitly; a skipped GPU test must not count as a passed GPU experiment. Pin APIs and add freshness entries when executable paths exist.

## Serial release units and dependencies

The units below are implementation milestones, not a promise to combine everything in one large PR. Each can be split into lecture and kit PRs if review size demands it; merge each before the next.

| Unit | Scope | Depends on | Status |
|---|---|---|---|
| P0 | This build plan and cross-links | Merged research blueprint | Merged in #45 |
| P1 | Master, decks 01–02 and L0; common fixture/result contracts | P0 | Merged in #46; CPU checks passed, student pilot pending |
| P2 | Decks 03–04 and L1; shared timing/profiling guidance | P1 | Merged in #48; CPU/analytical path, CUDA extension unverified |
| P3 | Decks 05–06 and L2 | P2 | Merged in #49; PyTorch CPU eager/compiler validation |
| P4 | Decks 07–08 and L3 | P3 | Implemented CPU references; CUDA source/build/run unverified |
| P5 | Decks 09–10 and L4 | P4 | Planned |
| P6 | Decks 11–12 and L5 | P5 | Planned |
| P7 | Decks 13–14 and L6 | P6 | Planned |
| P8 | Decks 15–16 and L7 | P7 | Planned |
| P9 | Decks 17–18 and L8 | P8 | Planned |
| P10 | Decks 19–20 and L9–L10 | P9 | Planned |
| P11 | Decks 21–22 and L11 | P10 | Planned |
| P12 | Decks 23–24 and L12 | P11 | Planned |
| P13 | Decks 25–26 and L13A/B | P12 | Planned |
| P14 | Decks 27–28 and L14 | P13 | Planned |
| P15 | Decks 29–30 and L15; revisit L10 | P14 | Planned |
| P16 | Capstone deck, package and complete-course pilot | P15 | Planned |

A unit with unavailable hardware may ship a clearly labeled analytical/reference path plus unverified GPU exercises, but must not be marked GPU-validated. Continue independent authoring while recording the exact missing validation. Do not publish fabricated results or advertise a runnable accelerated path that has never been checked.

## Definition of done per PR

- Source research and bibliography match every taught claim; code/API excerpts identify their version or inspection date.
- Decks have readable diagrams, worked examples, exercises and citations; inspect representative slides in a browser, including dense tables and code, and check overflow.
- All slides have continuous numbering, working navigation and no console errors. Master and README list every released deck with actual counts.
- Student instructions and actual kit files agree; correctness/reference/negative cases run on the declared environment. Record hardware skips and analytical-only scope visibly.
- Course consistency, site build/link and slide checks pass; changed executable labs run appropriate correctness tests and smoke experiments.
- PR description states scope, limitations, validation and dated prompt history. Check PR state, fetch/rebase on latest main before pushing, merge only after checks pass, and refresh main before the next unit.

## Final course pilot

Walk the student path from setup to capstone. Check that each topic has its deck and practical exercise, that references resolve, environments can be recreated, workloads fit the stated hardware and completion times fit the syllabus. Verify both training and inference objectives, agent cache behavior, and NVIDIA/AMD attribution/support boundaries. Revise pacing from pilot evidence; keep research findings, planned work and validated teaching artifacts visibly distinct.
