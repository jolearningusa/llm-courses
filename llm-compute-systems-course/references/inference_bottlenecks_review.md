# Inference bottlenecks and autoregressive decoding

Primary-source review, 2026-10-30. Supports [deck 36](../slides/36_inference_bottlenecks.html) and its [Chinese twin](../slides/zh/36_inference_bottlenecks.html). The course adds a connected diagnostic lesson because decks 01, 04, 15–21 and 30 already teach individual mechanisms but do not put serial token dependencies, actual GPU idle gaps and token-selection overhead on one map.

## Sequential generation does not imply an idle GPU

For ordinary autoregressive generation, the next distribution depends on previously selected tokens. A sequence therefore advances through dependent model/selection steps. Transformer training and prefill can process known input positions together within a layer using a causal mask; unknown future generated tokens cannot simply be supplied in advance. RNNs also carry temporal dependencies, but their recurrent hidden state and a transformer's KV state are different structures with different costs.

Within one decode step there is parallel work across tensor dimensions, heads and available sequences. Device dependencies can remain on the GPU; recurrence does not require copying every chosen token to the CPU. An apparently underused GPU may be moving weights/KV with little arithmetic, executing many short kernels, blocked on communication, or genuinely idle because the host has not submitted ready work. Each diagnosis requires different evidence.

[Speculative decoding](https://arxiv.org/html/2211.17192v2) (§§1–3, A.1, A.4) explicitly starts from serial target-model runs and uses proposed continuations plus verification/correction to advance multiple output positions. It does not make arbitrary unknown future tokens independent. Compare the whole draft/verify/state path and preserve the intended output distribution.

## Literature findings and their scope

| Primary source and examined sections | Finding relevant to the course | Practical limit |
|---|---|---|
| [Efficiently Scaling Transformer Inference, arXiv:2211.05102v1](https://arxiv.org/html/2211.05102v1), §§2–4, C | Batch, attention state and model partitioning trade latency against utilization/cost; long contexts add cache traffic. | Primarily PaLM/TPU-era experiments; mechanisms inform hypotheses, not a current GPU ranking. |
| [Splitwise, arXiv:2311.18677v2](https://arxiv.org/html/2311.18677v2), §§II–III, IV-C | Characterizes the differing prompt/token phases and their batching, memory and hardware behavior. | Its trace/model/hardware observations do not make every prefill compute-bound or every decode bandwidth-bound. |
| [Sarathi-Serve, arXiv:2403.02310v3](https://arxiv.org/html/2403.02310v3), §§3–5 | Low-token-count linear operations have low arithmetic intensity; batching moves them toward compute limits. Mixing prefill/decode creates latency interference. | The transition depends on shape, hardware, sharding and kernel implementation. |
| [How To Scale Your Model: inference](https://jax-ml.github.io/scaling-book/inference/), linear layers, attention, memory and batching sections | Derives compute/traffic/communication budgets and distinguishes throughput from per-sequence latency. | Idealized rates and reuse assumptions are bounds, not measured device performance. |
| [Look Ma, No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), workload and kernel-boundary analysis | A small-model, single-sequence case exposes execution-boundary overhead even in a weight-bandwidth-oriented workload. | A low-latency Llama-1B/H100 case; no general recommendation to replace a serving runtime with a megakernel. |
| [NVIDIA CUDA Graph best practices](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html), benefits/idle regions/synchronization | Correlate host launches and idle GPU regions; replay can reduce submission overhead. | A busy percentage alone cannot identify the bottleneck. Graph replay does not remove token dependencies or all device-side kernel-boundary costs. |
| [FlashInfer sorting-free sampling](https://flashinfer.ai/2025/03/10/sampling.html), algorithms and evaluation | Full vocabulary sorting is avoidable; rejection-based sampling is an existing optimization. | Sampling an equivalent distribution and returning an identical deterministic filtered tensor are different contracts. |
| [Qrita, arXiv:2602.01518v2](https://arxiv.org/html/2602.01518v2), §§3–5 | Pivot/truncation selection handles top-k/top-p, duplicates and fallback. Evaluates operator and end-to-end effects. | Paper §4.2 identifies batch-scaling limits; its engine/version/model results are not a universal current baseline. Recheck installed dispatch. |
| [GVR, arXiv:2604.22312v1](https://arxiv.org/html/2604.22312v1), §§2, 4–6, limitations | Exact top-k for sparse-attention context selection exploits temporal correlation in the studied DSA/Blackwell setting. | Separate from vocabulary sampling. Single-CTA design and context/prediction regimes limit transfer. |
| [TensorRT-LLM sampling and beam search](https://nvidia.github.io/TensorRT-LLM/latest/features/sampling.html), Torch Sampler/beam-search sections | Fused sampling and multiple-hypothesis beam execution are existing engine features with explicit semantics and compatibility limits. | Pin the release; neither feature coverage nor paper support proves an installed fast path. |

The key older papers remain useful for mechanism and experiment design; recent selection work shows that optimization shifts the limiting stage. No universal ordering of bottlenecks or fixed percentage breakdown follows from this review.

## Bottleneck map

| Regime | Candidate limiting resource | Evidence to collect | Intervention to test |
|---|---|---|---|
| Long prompt / enough parallel work | Dense compute or attention compute/IO | Shapes, operator times, achieved math/traffic rates | Supported fused attention, matrix kernels, chunking for interference |
| Small decode batch, large resident weights | Weight traffic / limited reuse | Linear-layer intensity, HBM traffic, batch sensitivity | Weight precision, better kernels, batching, eligible speculation |
| Long-context dense attention | KV reads and/or attention compute | Context/head/batch sweep, attention share, cache bytes | GQA/MLA-aware backend, supported KV precision; algorithm changes need quality gates |
| Many simultaneous/long sequences | KV capacity and admission | Live/reserved bytes, evictions, recomputation, OOM/queue events | Paging, prefix reuse, capacity-aware admission/offload |
| Small/fast models and many short kernels | Submission or device execution-boundary overhead | Host/device timeline and readiness gaps | Graph replay, fusion, supported persistent kernels |
| Large vocabulary or expensive logits processing | Output projection, filters, sampler | Time each separately with actual tensor layouts/parameters | Fused filters, exact selection or distribution-valid sampler |
| Beam width greater than one | More active hypotheses, selection and state management | Forward work, score selection, history/KV remapping | Supported sharing/indirection and batched candidate selection |
| Tensor/expert parallel deployment | Exposed collective/exchange/straggler time | Per-rank readiness, link traffic, maximum-rank completion | Placement, eligible communication/compute overlap, load balance |
| Saturated service / long prefill / tool waits | Queues, interference or application dependencies | TTFT/ITL, offered/completed work, whole-task timeline | Chunking, scheduling, admission and lifecycle policy |

A busy memory pipeline can coexist with low FLOP utilization. Conversely, a GPU-active percentage can be high while a workload wastes traffic or serializes tiny operations. Use the timeline plus relevant hardware counters; do not infer CPU starvation from one percentage.

## Worked budgets for teaching

These are constructed examples, not benchmark results. Assume an unsharded dense model with 8 billion parameters, BF16 weights and an effective 2 TB/s HBM stream. A single ideal read of 16 GB of weights takes 8 ms. For the same linear operation, processing more independent sequence rows can reuse that weight read and increase aggregate tokens/s, but does not divide one sequence's dependent token-step latency by batch size. Added compute, KV and scheduling can increase the iteration time.

For 32 layers, 8 KV heads, head dimension 128 and BF16, ordinary MHA/GQA logical KV is 128 KiB per stored token. A 32,768-token context occupies 4 GiB per sequence; eight such sequences occupy 32 GiB before block/allocator/scale overhead. Under an idealized full-cache read per decode iteration, that is also substantial traffic. Storage capacity and achieved read bandwidth are separate constraints; cache layout/reuse and attention variants change actual reads. Weight quantization does not shrink KV unless a separate KV recipe is used.

For fixed nonoverlapping work, a sampler taking an assumed 10% of elapsed time and becoming twice as fast yields 1/(0.9+0.1/2) ≈ 1.053× overall. Measure the original fraction and remeasure the full path; queueing and changed batching can break this simple model.

## Selection semantics that must stay separate

Vocabulary top-k selects candidate next tokens. Top-p keeps a probability-mass prefix and must specify ordering with top-k, normalization, ties and the cutoff rule. Beam search ranks sequence hypotheses using cumulative scores and possible length penalties, maintains finished candidates and updates ancestry. Sparse-attention top-k selects context positions inside the model, possibly per layer. MoE top-k selects experts and creates dispatch traffic. These share selection primitives but not correctness contracts or performance boundaries.

The Qrita paper's throughput gains and GVR's operator/TPOT gains establish examples of meaningful selection costs. They do not imply that top-k dominates ordinary decoding, that sorting is inherently impossible on GPUs, or that all sorting-free samplers return identical random draws. Keep output projection, filter computation, sampling RNG and host-visible token delivery separate in profiles.

## Course integration and student experiment

Use deck 36 after deck 17, revisit it after decks 19–21 and 24. It links existing foundations rather than replacing their detailed derivations. The [diagnosis worksheet](../labs/kit/inference_bottleneck_worksheet.md) extends L1/L8/L10 without changing their released CPU implementations. [P26](../projects/COURSE_APPENDIX.md#p26) provides a six-week selection/decoding-control project.

The minimum experiment fixes a model/engine/device and tests batch/concurrency, context length, eager/replay and supported sampling settings one factor at a time. Profile prefill, decode linear layers, attention, output projection, selection, host scheduling and communication at stated scopes. Use a separate unprofiled run for latency/goodput. Inspect actual dispatch and preserve quality/sampling semantics. CPU references and constructed budgets cannot validate GPU performance.
