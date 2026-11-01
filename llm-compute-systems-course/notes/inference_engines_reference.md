# Inference engines: reference matrix and experiment worksheet

Companion to [deck 35](../slides/35_inference_engines_compared.html) · [简体中文](zh/inference_engines_reference.md). Reviewed 2026-11-01 from project sources. This is optional reference material, not a speed ranking or an executed engine benchmark. Release lists and popularity counts are deliberately omitted; record the exact tested revision in an experiment.

## Candidate paths

| Project | Useful comparison question | Direct source |
|---|---|---|
| llama.cpp / llama-server | How do ggml backend, offload, slot policy and cache settings affect this model? | [Repository](https://github.com/ggml-org/llama.cpp), [server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) |
| Ollama | Which execution backend and defaults did the model-management/API layer select? | [Repository](https://github.com/ollama/ollama), [engine discussion](https://ollama.com/blog/multimodal-models) |
| MLX / MLX-LM | How do representation, residency and the Apple-silicon path affect the workload? | [MLX-LM](https://github.com/ml-explore/mlx-lm), [server limitations](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) |
| vLLM | Which scheduling, paged-KV and prefix-reuse path is active? | [Repository](https://github.com/vllm-project/vllm), [GGUF status](https://docs.vllm.ai/en/stable/features/quantization/gguf/) |
| SGLang | Does prefix-aware reuse help this arrival/history pattern? | [Repository](https://github.com/sgl-project/sglang), [documentation](https://docs.sglang.io/) |
| TensorRT-LLM | Which supported NVIDIA precision/kernel/runtime path fits the task? | [Repository](https://github.com/NVIDIA/TensorRT-LLM), [documentation](https://nvidia.github.io/TensorRT-LLM/) |

These are questions for candidates, not hard project boundaries. API compatibility must be checked by endpoint and feature; a common chat request does not guarantee identical sampling, tools, scheduling or numerical behavior.

## Configuration fields worth recording

| Concern | llama-server examples to inspect | Comparable question in another engine |
|---|---|---|
| Placement | `--gpu-layers`, `--split-mode`, `--tensor-split` | Device placement, sharding and transfer boundary |
| Concurrency | `--parallel`, batching configuration | Admission, live sequences and token budget |
| KV representation | `--cache-type-k`, `--cache-type-v` | Cache dtype, layout and quality checks |
| Prefix reuse | Request cache controls, reuse policy | Cache key, causal validity and hit rate |
| Cache tier | Host-cache configuration such as `--cache-ram` where supported | Retained bytes, transfer and recomputation costs |
| Speculation | Selected speculation type and draft model | Acceptance, verifier work and end-to-end benefit |

Use the installed version's help and record actual values rather than memorizing defaults. The reviewed server reference includes `none`, `layer`, `row` and **experimental `tensor`** split modes. Therefore “llama.cpp has no tensor-parallel path” is too broad. Conversely, a listed tensor split is not evidence of identical production scaling or prefill/decode-disaggregation behavior across engines.

GGUF is an extensible metadata/tensor format with memory-mapping and split-file support; see the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). It is not a guarantee that a loader implements every model/operator. Weight quantization, KV quantization and calibration are separate choices. Examples such as Q4_K_M describe recipes; their nominal bit widths do not make two representations numerically equivalent.

MLX unified memory changes host/device placement and explicit-copy choices; it does not remove bandwidth, synchronization or internal movement. MLX-LM's server documentation states that it is not recommended for production. vLLM's GGUF documentation marks that path experimental/under-optimized and discusses a plugin; check its version-specific limitations instead of assuming that loading a format implies the same optimized execution path. Do not infer an immutable Ollama backend from one release: record the tested model and selected backend.

## Experiment worksheet

1. Freeze model revision, tokenizer/chat template, task/quality criterion, sampling, prompt/output lengths and arrival process.
2. Record hardware, drivers, engine/backend revisions, weights/quantization/calibration, KV dtype, placement and all non-default options.
3. Run cold-prefix and shared-prefix cases with declared cache resets. Use a small concurrency sweep (for example 1, 8, 32 only if supported) under a fixed resource budget. Avoid silently treating cache warmup as an engine speedup.
4. Report TTFT, inter-token latency, completed useful tokens, tail/SLO attainment, rejected/canceled/failed requests and peak memory. Keep raw request records and startup/compile time separate.
5. Explain the result through batching, cache validity, kernel choice, representation and placement, using decks 15–17. State every variable that could not be matched.

If representations or quality cannot be matched, call the result a **configuration comparison**, not an isolated engine effect. That result can still inform a deployment decision under its declared constraints. The [L7](../labs/kit/handouts/lab07_cache.md) and [L8](../labs/kit/handouts/lab08_scheduling.md) handoffs in the deck give the existing lab path; no new engine installation or GPU result is claimed here.

[RadixAttention mechanism and historical evaluation](https://lmsys.org/blog/2024-01-17-sglang/): use the mechanism, not the historical speedups as a current ranking.
