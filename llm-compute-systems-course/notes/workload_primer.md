# Workload primer: just enough architecture and algorithms

Allocate roughly one lecture to this primer. The objective is to derive the tensors, data movement, synchronization and state that the computing stack must execute. Read [GQA](https://arxiv.org/abs/2305.13245), [DeepSeek-V2](https://arxiv.org/abs/2405.04434) and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) for concrete examples, rather than studying model architecture design in depth.

## Three execution phases

Training executes forward, backward and optimizer update, retaining or recomputing activations. A useful approximate dense-transformer training compute budget is `6 × parameters × training_tokens`; attention, embeddings, recomputation, routing and implementation details can require additional accounting. It is a planning approximation, not an exact instruction count.

Prefill computes a prompt's states in parallel over its tokens, often making large GEMMs efficient. Decode generates the next token for each active sequence, often exposing small GEMMs, weight/KV traffic and launch/communication latency. Neither phase has one universal bottleneck: batch, context, architecture, placement and kernel choices can change it.

For GEMM `(M,K) × (K,N)`, start with `2MKN` floating-point operations and the bytes of inputs, output and any intermediates. Then inspect actual cache behavior, fusion and reuse. A first lower bound is `max(FLOPs / achieved_compute_rate, bytes / achieved_bandwidth)` within a phase where overlap is possible. Sequential dependencies and launches still contribute time.

## What different optimizations actually save

| Mechanism | Resource affected | Semantics and limits |
|---|---|---|
| KV caching | Repeated K/V projections and recomputation for already processed tokens | Exact reuse under valid causal/cache conditions; full-attention decode still attends to all retained keys |
| Prefix caching | Prefill work shared across compatible requests | Requires matching tokens, positions, model/adapter and cache configuration; cache storage, lookup and eviction have costs |
| FlashAttention | HBM reads/writes and large materialized attention intermediates | Computes exact dense attention in the algorithmic sense, subject to floating-point effects; does not remove its quadratic prefill arithmetic |
| PagedAttention | KV allocation waste and duplication | Logical block mapping; does not inherently reduce the mathematical attention range |
| MQA/GQA | KV heads, cache size and associated bandwidth | An architectural choice; checkpoint conversion/uptraining and quality matter, rather than a universal inference switch |
| MLA | Cached representation via compressed latent state | Architecture-specific projections and positional treatment; use its actual cache formula and kernels |
| KV quantization | Stored cache bytes and transfer volume | Adds scales, quantize/dequantize work and numerical error; verify quality and supported kernels |
| Sliding-window / sparse attention | Attended pairs and sometimes retained state | Changes attention connectivity; requires architecture support or approximation/quality evaluation |
| Linear attention / state-space recurrence | Context-dependent attention work/state | Different computation/architecture; Mamba is a selective state-space model, not simply a drop-in exact-attention kernel |
| Speculative decoding | Serial target-model steps per accepted token | Correct verification/correction can preserve the target distribution; draft/verification overhead can outweigh savings |
| Fusion / CUDA Graph replay | Intermediate traffic, launch and host scheduling overhead | Preserve computation only if dependencies/aliasing/state semantics remain correct; shape and buffer constraints matter |

Sources: [FlashAttention](https://arxiv.org/abs/2205.14135), [PagedAttention](https://arxiv.org/abs/2309.06180), [Mistral 7B](https://arxiv.org/abs/2310.06825), [Mamba](https://arxiv.org/abs/2312.00752), [speculative decoding](https://arxiv.org/abs/2211.17192). These are separate mechanisms, not alternative names for caching.

## A concrete KV budget

For ordinary MHA/GQA with identical key/value head dimensions, total logical KV bytes across a batch are approximately:

```text
2 × layers × KV_heads × head_dim × bytes_per_element × sum(retained_tokens_per_sequence)
```

Example: 32 layers, 8 KV heads, head dimension 128 and BF16 storage use 131,072 bytes (128 KiB) per retained token. A single 32,768-token sequence uses 4 GiB for its logical KV tensors. Replacing 8 KV heads with 32 increases this to 16 GiB. This is a constructed GQA/MHA example, not a claim about a particular checkpoint.

Add paging slack, metadata, scales, workspaces, allocator reserve and weights before estimating device fit. Per-GPU memory depends on actual partitioning and replication; division by tensor-parallel degree is not always valid. MLA, hybrid layers, unequal key/value dimensions and windowed caches need different formulas. Paged allocation does not shrink the bytes of live logical KV data.

## MoE in a computation trace

An MoE block commonly routes each token to a small top-k subset of experts. The systems path is **router → token permutation/packing → dispatch communication → expert GEMMs → combine communication → unpermutation/reduction**. Shared experts and implementation variants add work. Read the [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) with [DeepEP](https://github.com/deepseek-ai/DeepEP) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).

Separate total parameters, active parameters, resident weights and weights actually read. Activating fewer experts reduces per-token arithmetic relative to evaluating all experts, but all resident experts still consume capacity. Small or uneven token assignments can underfill GEMMs; routing skew creates stragglers; expert parallelism introduces communication. Fewer FLOPs can therefore coexist with worse latency.

Students vary tokens per expert and routing skew at constant total input tokens, then inspect dispatch, compute, combine and the longest expert path separately. Training adds saved activations, gradients and optimizer state; inference adds latency constraints and persistent KV state.

## Training state accounting

Write down parameter, gradient, optimizer and optional master-copy dtypes separately. One illustrative mixed-precision Adam configuration uses 2 bytes for parameters, 2 for gradients, 4 for an FP32 master copy and 8 for two FP32 moments: 16 bytes per parameter before activations and buffers. This is not a universal PyTorch default. BF16 training without a separate master copy or with different gradient storage has a different total.

ZeRO/FSDP shard different parts of this state. They do not remove transient all-gather buffers, activations, workspaces or communication time. Activation checkpointing trades memory for recomputation; distributed checkpointing writes recoverable training state to storage. These similarly named mechanisms solve different problems. See [ZeRO](https://arxiv.org/abs/1910.02054), [FSDP](https://arxiv.org/abs/2304.11277) and [selective recomputation](https://arxiv.org/abs/2205.05198).

## Exit questions

1. What remains linear in context length in one full-attention decode step after KV caching?
2. Which of FlashAttention, PagedAttention and GQA can be adopted without changing the logical attention function for a fixed checkpoint?
3. Why does reducing active MoE FLOPs not guarantee the model fits in one GPU?
4. Why can a quantized operation move fewer bytes and still take longer?
5. Which measurements distinguish a capacity problem from a bandwidth or scheduling problem?
