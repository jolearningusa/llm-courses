# Syllabus: Efficient LLM Computing Systems

Sixteen weeks, two 75–90 minute lectures and approximately 5–8 hours of practical work per week. Timing is a planning estimate to revise after a pilot. Full lab specifications are in [LAB_PLAN.md](labs/LAB_PLAN.md); the [kit README](labs/kit/) identifies currently released paths and their validation limits.

## Teaching contract

The subject is the computing system. A small decoder and a small MoE layer provide repeatable tensor workloads; students do not need to train a competitive language model. Architecture/algorithm context occupies approximately one lecture plus short explanations at the point of use. A dense workload and an expert-routed workload recur throughout the course.

Every optimization must answer: which resource is limiting, which mechanism changes that limit, which correctness conditions hold, and whether the job-level objective improves. Training uses useful tokens/s and time to a fixed validation-loss target where feasible. Serving uses accepted output tokens/s at stated latency and quality constraints. Energy and cost are evaluated only when measurement boundaries and pricing assumptions are supplied.

The [build plan](BUILD_PLAN.md) turns every lecture below into a named deck and paired practical exercise, delivered as sequential merged PRs.

## Course parts

The core of the course is inference: Parts I–V follow one request's computation from the workload and a single GPU to a serving engine and a deployed program. Training at scale comes last, as Part VI, and reuses the same kernels, precision formats and measurement discipline.

- **Part I · Foundations: workloads and measurement** (weeks 1–2; decks 01–04; L0–L1). What work, data movement and memory capacity does this job require, and how do we measure it honestly?
- **Part II · Single-GPU execution: PyTorch, compilers and kernels** (weeks 3–6; decks 05–12; L2–L5). How does a tensor program become efficient work on one device?
- **Part III · Numerics and quantization** (week 7; decks 13–14; L6). Which formats, scales and accumulations keep a model correct while moving fewer bytes?
- **Part IV · Inference serving** (weeks 8–11; decks 15–21; L7–L10). How do we deliver useful tokens under latency, memory and load constraints?
- **Part V · Deployment and platforms** (weeks 11–12; decks 22–24; L11, L15A). How do we package a program for a target, and what does the target platform provide?
- **Part VI · Training at scale** (weeks 13–15; decks 25–30; L12–L14, L6B, L15B). How do we fit, synchronize and recover a training job, and does a component gain improve the whole job?
- **Capstone** (week 16; deck 31).

Supplements: deck 33 (programming models beyond CUDA) extends Part II after deck 07; deck 32 (optimization decisions and automated search) follows Part II and is revisited after Part V.

## Weekly schedule

| Week | Lecture A | Lecture B | Practical evidence and required reading |
|---|---|---|---|
| **Part I** | **Foundations: workloads and measurement** | | |
| 1 | **The execution stack.** Training forward/backward/update versus prefill/decode; rooflines, latency and capacity; tensor shapes and bytes. | **A workload primer.** Dense/MoE, GQA/MLA, cache reuse, exact versus restricted attention. Inspect computation rather than design a new model. | L0: derive budgets and predict bottleneck transitions. Read the workload primer and *How To Scale Your Model*, roofline chapter. Skim DeepSeek-V2/V3 for workload structure. |
| 2 | **Measurement.** Wall time, CUDA events, asynchronous execution, warmup, compilation, synchronization, allocator state and profiler perturbation. | **Host and OS.** Python overhead, threads, CPU affinity, NUMA, pinned memory, DMA, page faults, HMM and data/input pipelines. | L1: diagnose a CPU-to-GPU starvation trace. Required: CUDA Best Practices and Nsight profiling guidance; Linux HMM is a focused reference. |
| **Part II** | **Single-GPU execution: PyTorch, compilers and kernels** | | |
| 3 | **PyTorch execution.** Tensor storage, strides, views, dispatcher, ATen and library calls; autograd tape and saved tensors; optimizer execution. | **Compiler path.** Dynamo capture/guards → FX → AOTAutograd where applicable → Inductor scheduling/code generation → kernels/libraries. Distinguish graph breaks from recompilation. | L2: compare eager/compiled forward and training steps with shape changes. Required: PyTorch 2 paper, official compiler docs. |
| 4 | **CUDA SIMT.** Grids/blocks/warps, coalescing, divergence, registers, shared memory and occupancy. | **Reductions and integration.** Stable softmax/RMSNorm, warp collectives, barriers, race detection, custom PyTorch operator metadata and backward support. | L3: implement a reduction, inspect traffic, validate tails/strides and gradients if used for training. Required: CUDA Programming Guide; Triton fused-softmax tutorial as a comparison. |
| 5 | **GEMM engineering.** Arithmetic intensity, tiling, layouts, bank conflicts, reuse, Tensor Cores, epilogues, occupancy versus registers. | **Asynchronous kernels.** Pipelines, double buffering, TMA, warp specialization and persistent scheduling. Compare CUDA C++, Triton, CuTe DSL and CUDA Tile. | L4: shape-dependent GEMM benchmark against an optimized library. Required: CUTLASS documentation/examples. Advanced: CUDA Tile tutorial; Rubin TMA changes. |
| 6 | **Attention execution.** Online softmax, tiled exact attention, saved intermediates, recomputation in backward, masking and numerical stability. | **Hardware/algorithm co-design.** FlashAttention 1→2→3→4; load balance, matmul/softmax asymmetry, specialized decode kernels. | L5: reference exact attention and a tiled path. Required: FlashAttention and FA2; FA3/FA4 selected sections. Treat historical benchmark comparisons as dated experiments. |
| **Part III** | **Numerics and quantization** | | |
| 7 | **Numerical formats.** BF16/FP16, FP32 accumulation, FP8, INT8/INT4 and block-scaled formats; range/outliers/scales and quantization cost. | **Quantization for serving.** Transformer Engine and block-scaled formats, TorchAO PTQ, weight-only versus weight-and-activation quantization, KV quantization, packed layouts, kernel support and serving quality gates. | L6A/C: byte, error and memory comparison with an inference quality gate. Required: TorchAO and official TE docs; AWQ/SmoothQuant as contrasting recipes. The training-precision exercise (L6B) is done with deck 25 in week 13. |
| **Part IV** | **Inference serving** | | |
| 8 | **Serving runtime.** Iteration scheduling, continuous batching, token budgets, cancellation, admission, memory allocation and preemption. | **KV state.** PagedAttention, block tables, copy-on-write, prefix caching, eviction and attention-backend dispatch. | L7: trace a request through nano-vLLM and vLLM or SGLang. Required: Orca and PagedAttention; FlashInfer runtime sections. |
| 9 | **Latency-throughput tradeoffs.** Long-prefill interference, chunked prefill, scheduling fairness, queueing and SLO goodput. | **Disaggregation.** Separate prefill/decode, KV transfer, placement, cache-aware routing, CPU/storage tiers and transfer/recompute break-even. | L8: compare colocated and split serving under fixed total resources. Required: Sarathi-Serve and DistServe; Mooncake as extension; inspect Dynamo/NIXL/LMCache roles. |
| 10 | **MoE as an inference workload.** Routing/permutation, grouped GEMM, dispatch/combine, expert-parallel all-to-all, expert imbalance and small per-expert decode batches. | **Decode acceleration.** Draft/verify/correct speculative decoding, acceptance and draft cost; request-level versus kernel-level gains. | L9: skewed routing and communication/compute overlap. L10A: speculative benefit across acceptance and load. Required: DeepSeek-V3, DeepEP/DeepGEMM; Leviathan et al. |
| 11 | **Inference beneath AgentOS.** Repeated calls, tool waits, KV TTL, cache-aware routing, invalidation and task latency; distinguish tensor reuse from semantic response caching. | **Part V · Deploying compiled programs.** `torch.export`, shape contracts, AOTInductor `.pt2` packages, startup versus steady state, custom-op support. | L10B: retain/offload/recompute policies under several loads; no unconditional claim that smaller work is faster. L11A: export and AOTInductor deployment. Required: Autellix/Continuum selections, the agentic caching note and official AOTInductor docs. |
| **Part V** | **Deployment and platforms** | | |
| 12 | **Edge/host deployment.** ExecuTorch lowering/delegates, `.pte` programs and optional `.ptd` data; backend boundaries, memory planning and TorchAO compatibility. | **Vera Rubin integration.** CPU–GPU–HBM–NVLink–NIC topology; relate kernels, precision and placement to source-documented mechanisms. | L11B: exported/target-lowered execution, CPU targets first; CUDA delegates need explicit compatibility checks. L15A: Rubin/NVIDIA/AMD architecture dossier. Required: official ExecuTorch docs, Rubin technical architecture article and platform documentation. No rack required for the analytical path. |
| **Part VI** | **Training at scale** | | |
| 13 | **Training precision and distributed state.** AMP/loss scaling, QAT and training quality gates; DDP gradient synchronization, optimizer/gradient/parameter bytes, ZeRO stages and FSDP2 all-gather/reduce-scatter timelines. | **Communication mechanics.** Collectives, latency/bandwidth models, buckets, stream dependencies, overlap, stragglers and topology placement. | L6B: autocast loss/gradient comparison. L12: DDP versus FSDP2 with fixed effective batch and peak-memory measurement. Required: ZeRO, FSDP paper, current FSDP2 tutorial and NCCL guidance. |
| 14 | **Parallelism composition.** Tensor, pipeline, sequence and context parallelism; microbatch schedules, activation recomputation and pipeline bubbles. | **Training implementation study.** Trace TorchTitan and Megatron Core configuration → DeviceMesh/sharding → kernels → collectives. | L13: choose a parallelism plan under capacity/interconnect constraints. Required: Megatron large-scale training paper and activation recomputation paper. Inspect upstream code at a recorded commit. |
| 15 | **Training operations.** Dataset ingestion, asynchronous distributed checkpoints, optimizer/RNG/data-cursor restoration, failure recovery and useful goodput. | **Whole-job efficiency.** Power, useful throughput, checkpoint stalls, queueing, utilization and cost; evaluate vendor claims with explicit baselines. | L14: interruption/resume and useful-throughput report. L15B: reconcile component gains with the job-level outcome. Required: distributed-checkpoint docs. |
| **Capstone** | | | |
| 16 | **Capstone review.** Training systems presentations and trace walkthroughs. | **Capstone review.** Serving/deployment presentations; negative results, portability and unresolved limits. | Submit a reproducible artifact, fixed workload description, causal ablations and report. |

## AMD comparison thread

AMD ROCm is integrated across kernel, collective and serving comparisons, using the [AMD stack note](notes/amd_stack.md). Compare HIP/CK/AITER and hipBLASLt with CUDA/CUTLASS and library paths, and RCCL/rocSHMEM with communication responsibilities. Hardware access is optional; factual library roles and portability limits are core content.

## Shared experiment design

Use a small dense decoder first, with fixed weights/seeds and a fixed token corpus; then substitute an MoE feed-forward block and vary expert count, top-k, routing skew and tokens per expert. For training, keep the effective batch, optimizer semantics, data order and validation set constant across system comparisons. For inference, fix tokenizer, prompt/output lengths, sampling settings, arrival process and quality checks. A performance experiment may use synthetic tensors, but label it as a kernel or system microbenchmark rather than model-quality evidence.

Three recurring shapes prevent overfitting the course to one regime: a high-token-count training/prefill GEMM, a small-batch decode GEMM and an imbalanced MoE grouped GEMM. Add short/long contexts and changing batch sizes. Explain when compute, memory, launch overhead, communication or queueing becomes dominant.

## What students build

The proposed lab sequence produces a budget model, measurement harness, PyTorch compiler experiment, reduction/GEMM/attention kernels, low-precision experiment, distributed training configuration and recovery test, serving experiment, and deployment comparison. These are bounded components rather than a new production training framework. Use TorchTitan/Megatron and vLLM/SGLang as code-reading and integration targets; students implement enough of each mechanism to explain its behavior.

## Capstone

Choose one objective: improve a training job's useful throughput/recovery, improve serving goodput at declared TTFT/ITL targets, or improve a deployed program's startup/memory/latency. The proposal must name the bottleneck and a counterfactual experiment. A kernel-only project must show where the kernel sits in a job and how its time fraction limits end-to-end benefit.

Deliver a baseline, changed implementation/configuration, correctness/quality checks, environment and hardware manifest, raw measurements and a 4–6 page report. Include at least two causal ablations, repeated trials and one workload where the method is neutral or worse. For training, demonstrate loss/gradient behavior and account for checkpoint/recovery time. For serving, report offered load, completed/rejected/canceled requests, tail latency and SLO attainment together.

Suggested rubric: correctness/quality 25%, experimental validity 25%, causal systems explanation 25%, reproducibility 15%, performance/resource improvement 10%. Negative results can earn full explanation credit. No credit for presenting analytical Rubin projections as measurements.

## First paper discussion

Compare FlashAttention, PagedAttention and GQA. For each, draw the dataflow, identify what is reused or moved, state whether model semantics/weights change, and identify which resource gets smaller. Explain why all three may coexist in the same serving engine. This discussion establishes the vocabulary used throughout the course.

## Supplement: optimization decisions and automated search

Use released deck 32 and [the technique map](notes/optimization_techniques.md) after Part II, then revisit after Part V. The supplement distinguishes platform, library, graph compiler and kernel DSL; compares existing vLLM/TileLang tuning and FlashInfer-Bench; and asks students to measure final performance **and search cost** with held-out workloads. This adds a cross-topic section without replacing the training/inference objectives or implying that every target has been executed.
