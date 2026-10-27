# Lab specifications

This document specifies the full assignments. The [released kit](kit/) currently provides L0 and the CPU/analytical path for L1; later kits and GPU validation remain planned. Proposed timing is 5–8 hours for a major lab, subject to pilot calibration. L0 and L15 are analytical work; selected shorter exercises may be combined. Students should not need to install every upstream stack.

## Common protocol

Every release includes a pinned environment, a CPU/reference path where meaningful, target-hardware requirements, fixtures, correctness checks, workload manifest and machine-readable results. Use the [report template](report_template.md). GPU timing requires correct synchronization or device events, warmup, repeated trials and a separate cold-start report. Record actual library/backend dispatch. Use profiler runs for diagnosis and unprofiled runs for primary timing.

Kernel tests include irregular dimensions, masks/strides where supported, dtype-appropriate error, and memory/race diagnostics. Training paths validate gradients and short loss behavior. Multi-GPU paths validate rank consistency and failure behavior. Do not demand a fixed speedup; require a causal explanation and an adverse workload.

## L0 — Predict the resource budget (week 1; CPU)

Build a spreadsheet or small calculator for GEMM FLOPs/bytes, MHA/GQA KV state and training-state dtypes/sharding. Reproduce the 4 GiB example in the workload primer, then add allocator/workspace headroom and explain logical versus physical memory. Sweep batch/context size to predict when a model runs out of memory and when arithmetic intensity changes. Acceptance: units and all assumptions visible; MLA/MoE use their actual representation instead of a dense formula applied blindly. Hand in budget tables and three falsifiable predictions.

## L1 — Explain a slow timeline (week 2; one GPU or supplied trace)

Run or inspect a short training and inference trace. Introduce one bottleneck at a time: CPU data preparation, pageable transfer, launch gaps or allocation churn. Change affinity/prefetching/pinning/stream use only when supported and justified. Measure end-to-end time alongside GPU events and distinguish allocated/reserved memory. Acceptance: identify the actual limiting phase and show that the intervention changes the expected trace signature. Supplied-trace students submit analysis, not locally measured GPU claims.

## L2 — Follow PyTorch into the compiler (week 3; CPU or GPU)

Use a small linear/normalization block for eager and compiled forward/training. Inspect FX/generated execution, graph breaks, guards and saved tensors. Vary batch/sequence lengths and compare specialization with dynamic shapes or bucketing. Acceptance: output/gradient checks pass; cold compilation and steady time are separate; students identify one compiled and one library-dispatched operation. Explain a case where compilation is neutral or worse. Source: official compiler documentation.

## L3 — Implement a reduction (week 4; CUDA GPU)

Implement RMSNorm or stable softmax in CUDA C++, then compare a supported Triton path. Begin with a simple correct version; change coalescing, reduction organization and fusion one at a time. Validate tails and declared stride support; reject unsupported inputs clearly. If used in training, supply and check backward. Acceptance: no sanitizer errors, justified tolerances, measured bytes/time and one profile-supported explanation of register, bandwidth or synchronization cost.

## L4 — Engineer GEMM for three regimes (week 5; suitable GPU)

Benchmark a tiled implementation and cuBLASLt baseline on training/prefill, decode and grouped-MoE shapes. Inspect layout, tensor instructions, epilogues and pipeline behavior; compare CUTLASS/CuTe or CUDA Tile as an advanced path. Acceptance: actual compute dtype/accumulation reported, optimized-library baseline included, allocation excluded or explicitly included, and shape-dependent results explained. AMD option: hipBLASLt/CK on supported hardware. A source port without correctness/performance validation is not completion.

## L5 — Exact attention and ragged scheduling (week 6; GPU, reference on CPU)

Implement a reference and tiled online-softmax forward path; inspect backward/recomputation and compare supported library attention. Use causal/noncausal masks, variable lengths and at least two head dimensions. Hold semantics fixed before experimenting with approximate/windowed variants. Acceptance: dtype-appropriate reference comparison and scaling plots for memory and time; distinguish FLOP complexity from intermediate storage. Extension: inspect FA3/FA4 pipelines on their intended hardware or analyze supplied traces.

## L6 — Low precision with a numerical budget (week 7; hardware-dependent)

Choose one supported TorchAO or Transformer Engine recipe; compare baseline with transformed storage/execution. Measure scale conversion, packed storage, kernel and whole-block execution. Training extension: evaluate gradients and a fixed short loss trajectory; inference extension: evaluate a fixed quality set. Acceptance: calibration data separate from evaluation, scale/accumulation choices recorded, quality threshold declared before performance results. Include an adverse small shape. An unsupported fast dtype must produce an explicit skip, not an undisclosed fallback.

## L7 — DDP versus FSDP2 (week 8; 2–8 GPUs)

Train the same small workload with identical effective batch and optimizer semantics. Predict persistent state and transient all-gather memory, then measure peak allocation and collective timelines. Change sharding boundaries/buckets/prefetching or overlap one at a time. Acceptance: equivalent short-run training behavior, accounting of synchronization and communication, and memory savings reconciled with step time. CPU simulation is a separate analytical submission. AMD option: supported PyTorch/ROCm with RCCL, recording actual backend.

## L8 — Choose and justify parallelism (week 9; multi-GPU or analytical trace)

Use TorchTitan or Megatron Core at a recorded compatible commit. Compare two feasible combinations of TP/PP/data sharding; inspect sequence/context parallelism and activation checkpointing as extensions. Keep global batch/tokens constant and separate warmup/compile. Acceptance: rank placement, microbatches, pipeline bubbles, communication and activation peaks documented; explain why one plan wins under the available topology. Do not claim FSDP and tensor parallelism partition computation identically.

## L9 — MoE traffic under imbalance (week 10; GPU, multi-GPU extension)

Use identical token count and vary expert routing skew. Instrument permutation, dispatch, grouped GEMM, combine and unpermutation. Examine DeepGEMM/DeepEP or AITER implementation choices appropriate to the device. Acceptance: routing and recombination match a reference, hottest-expert delay is visible, total/resident/active weights are separated, and communication cost is measured where distributed. Compare uniform and skewed routing; do not substitute balanced synthetic routing for the final conclusion.

## L10 — Checkpoint and recover useful training (weeks 10–15; CPU demonstration, multi-GPU validation)

Compare synchronous and supported asynchronous distributed checkpointing. Inject an ordinary controlled process interruption, restore model/optimizer/scheduler/RNG and data position, and continue a fixed workload. Acceptance: restore correctness and resumed progression established; report checkpoint pause, background work interference, storage time, recovery time and tokens lost/repeated. Separate activation checkpointing from persisted checkpoints. Keep a protected baseline checkpoint for the experiment.

## L11 — Cache anatomy across engines (week 11; CPU llama.cpp path or GPU serving)

Read [agentic_caching.md](../notes/agentic_caching.md), then compare two available engines from vLLM, llama.cpp and SGLang. Use cold, shared-prefix, early-mismatch, late-mismatch and memory-pressure workloads. Trace hash blocks/radix paths/slots according to the engine; distinguish live state, reusable prefix state and persistence. Acceptance: avoided prompt tokens and cache memory measured, exact version and settings recorded, no claim of cross-engine portable state. Inspect more conversations than available slots and a model/adapter-change invalidation case. Never score caching by latency alone without confirming that the intended request was evaluated.

## L12 — Schedule prefill, decode and KV transfer (week 12; 2+ GPUs for measured split serving)

Compare baseline scheduling, chunked prefill and supported prefill/decode separation on the same total device budget. Sweep arrival rate, context and prefix sharing; add transfer latency/bandwidth measurements. Acceptance: open-loop offered load, achieved load, TTFT/ITL tails, rejects/timeouts and SLO goodput reported together. Include an interference-dominated case and a transfer-dominated case. Analytical runs use measured/supplied phase costs and are labeled simulation. Do not compare two split GPUs with one colocated GPU as if resources matched.

## L13 — Speculation and agent waits (week 13; one GPU, trace simulation option)

First evaluate draft/verify/correct decoding with fixed target semantics; sweep draft cost and acceptance. Then replay model → tool wait → append result → model traces, comparing immediate KV eviction, retention and a bounded TTL/offload policy. Acceptance: correct speculative acceptance/correction, recorded draft overhead, cache byte-seconds and end-to-end task latency. Include bursty arrivals and canceled/unresumed tasks. Simulation must use explicit costs and cannot establish hardware speedup. Extension: cache-aware routing with queue imbalance.

## L14 — Export and deploy a computation (week 14; CPU first, GPU/edge extensions)

Use identical weights/input for eager, compiled, exported-and-reloaded, AOTInductor and one supported ExecuTorch target. Begin with a small block so toolchain effects are inspectable before extending to an LLM. Measure export/build, fresh-process load/first run, steady latency, memory and artifact size. Acceptance: capture/shape contracts and backend delegation verified; numerical checks pass; unsupported paths are explicit. Add one shape outside the contract and explain the result. A `.pt2` filename alone must not be used to identify its producer/loader.

## L15 — Rubin and AMD portability dossier (week 15; analytical core, hardware extension)

Draw a training and an agentic serving dataflow on the documented Rubin topology. Map each bottleneck to an actual NVIDIA software layer. Compare the corresponding AMD responsibility and identify a supported target path; list reused, retuned and replaced components. Acceptance: all hardware facts/source dates cited, vendor claims separated from measurements, software compatibility unverified until tested. Include both retain/offload/recompute and a collective/compute example. Hardware access is an extension, not a prerequisite.

## Deck and kit production mapping

The [slide and lab build plan](../BUILD_PLAN.md) maps every lecture to its paired exercise and defines the kit files, environment paths, validation gates and serial implementation order. This document remains the experiment specification; the build plan tracks production.

## Release order

Implement shared measurement/reference utilities and L0–L2 first; then kernel labs L3–L6, distributed training L7–L10, serving/cache L11–L13 and deployment/integration L14–L15. Ship bounded lab kits in separate PRs following [the roadmap](../ROADMAP.md). Each release must have a named tested configuration and clear skip behavior for unavailable hardware.
