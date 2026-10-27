# Inference and caching beneath an AgentOS

An agent alternates model execution, tool waits and further model calls, often branching or delegating. The system objective is useful task completion under latency, capacity and quality constraints. A per-request tokens/s benchmark alone misses the memory retained during waits and the recomputation caused by eviction.

## Disambiguate “GPT mem-cache”

That phrase does not identify one standard mechanism. Teach these distinct cache layers before comparing engines:

| Layer | Cached object | Saves | Main correctness question |
|---|---|---|---|
| Application response cache | A completed answer, keyed exactly or by semantic similarity | A whole inference call on a valid hit | Is the answer valid for this user, context, freshness and tool state? |
| Prompt/prefix cache | Previously computed state for a matching token prefix | Repeated prefill of that prefix | Is this the same computation, including model, adapter, positions and modalities? |
| Live per-sequence KV | K/V or architecture-specific recurrent/latent state | Recomputing prior state at every new token | Does the state correspond to the current sequence and attention contract? |
| Offloaded/persistent state | Engine-specific serialized or transferred sequence/cache state | Recomputing after a wait, handoff or restart, if restoration is compatible | Can this target/runtime restore it correctly, and is transfer cheaper? |
| Compiled-code/graph cache | Generated kernels, compiler results or replayable execution | Recompilation or repeated launch setup | Do guards, buffers, shapes and runtime assumptions still hold? |
| Model/OS storage cache | Weight pages or file data | Some disk/transfer work | Is the weight version right, and what data movement remains? |

[GPTCache](https://github.com/zilliztech/GPTCache) is an example of semantic response caching; it is not the KV allocator in vLLM or llama.cpp. Generic key/value storage such as memcached is another abstraction: storing arbitrary bytes does not implement tensor compatibility, attention semantics or cache ownership. Agent long-term text/vector memory is also distinct from serving state.

## Mechanisms in three inference stacks

| Stack | Source-documented mechanism | What to inspect in the lab |
|---|---|---|
| vLLM | Prefix blocks identified by hashes incorporating parent prefix, token IDs and extra identity such as adapters/modal inputs; full blocks are cached. The manager maintains block references and eviction/free structures. | Logical prefix match → existing block references → suffix allocation → free/eviction; inspect the installed version's key construction and metrics |
| SGLang | The SGLang paper's RadixAttention organizes prefix reuse with a radix tree. | Shared path versus divergent suffix; branch lifetime, eviction and the actual current cache backend |
| llama.cpp server | `cache_prompt` reuses a compatible common prefix and evaluates the new suffix. Documentation also describes a RAM prompt-cache pool, idle-slot behavior and slot save/restore. | Distinguish live-slot reuse, RAM-backed state reuse and disk persistence; inspect slot affinity and memory/restore costs |

Primary sources: [vLLM design](https://docs.vllm.ai/en/latest/design/prefix_caching/), [SGLang paper](https://arxiv.org/abs/2312.07104), [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). Flags, defaults and eligible model/cache types are version-specific. Do not teach a shared portable KV file format: these engines use different state and execution representations.

The llama.cpp docs warn that cache reuse can change bitwise output behavior across batching/backend choices. Validate numerical behavior and sampling expectations rather than require identical sampled strings as the sole correctness test. A GGUF model artifact stores a model for the runtime; it is not synonymous with a saved session's KV state.

## A block-cache mental model

This is teaching pseudocode, not an implementation of any engine's complete key:

```text
identity = model_revision + adapter + positions/mask + dtype/layout + modality + trust_domain
parent = empty_prefix
for each eligible token block:
    key = identify(parent, tokens_in_block, identity)
    if a compatible block exists:
        acquire reference to block
    else:
        evaluate missing prefix work and store resulting state
    parent = key
append and compute the uncached suffix
release live references when execution no longer owns them
evict only state that the policy and ownership rules permit
```

A matching suffix after an earlier mismatch is not generally reusable as if it had the same causal history. A changed system prompt, chat template or inserted tool output can alter all following states. Keep stable shared prefixes in a reproducible format, but do not reorder content in a way that changes intended model behavior merely to raise a hit rate.

## A multi-turn agent trace

```text
shared instructions + tools + user input
    → prefill → decode tool request
    → external tool wait
    → append tool output → prefill new suffix → decode
    → branch/delegate → join or discard → next turn
```

During the wait, the scheduler chooses retain, offload or evict/recompute. Long waits waste scarce HBM if all state is retained; short waits may make eviction wasteful. Prefix-aware routing helps only when locality gains outweigh queue imbalance. Branch sharing needs correct lifetimes and copy-on-write behavior where applicable. Cancellation should release ownership without corrupting a still-shared prefix.

Read [Autellix](https://arxiv.org/abs/2502.13965) for program-level scheduling and [Continuum](https://arxiv.org/abs/2511.02230) for multi-turn scheduling with KV time-to-live. They motivate the course experiment; do not assume their policies are current defaults in vLLM or llama.cpp.

## Retain, offload or recompute

For each wait, estimate:

```text
recompute cost = measured prefill time for the missing prefix
restore cost = transfer time + lookup/registration/format/queue overhead
retention cost = state bytes × wait duration, plus displaced work or queueing
```

Byte-seconds are a useful accounting measure, not a price by themselves. Offload is attractive only when compatible restore costs less than the work/memory pressure it avoids. Tail behavior, burstiness, transfer contention and the probability of a resumed task affect the decision. Caching more state is not always the highest-goodput policy.

## Cache validity, isolation and measurement

Treat model/adapter changes, tool-result freshness and trust domains explicitly. A response cache must invalidate stale external facts even if two prompts are semantically close. A tensor cache must not reuse incompatible state. vLLM documents optional cache salting for controlled sharing domains; that mechanism is not a complete application authorization policy. Cross-user hit rates can also reveal information through timing, so specify who may share state.

Report matched tokens and avoided prefill tokens, not only a request hit fraction. Also record live/idle KV bytes, cache byte-seconds, evicted/recomputed tokens, transfer bytes/time, TTFT after tool return, per-token latency, task completion time and quality/success. Run cold cache, warm shared prefix, early-prefix mismatch, late mismatch, eviction pressure and more conversations than slots. Separate process warmup and compiler caches from tensor-cache effects.

## Connection to Vera Rubin

Rubin's agentic emphasis makes this a useful whole-system example: a fast decode kernel can still leave tasks waiting on CPU orchestration, external tools, KV movement or queueing. [NVIDIA's post-training discussion](https://blogs.nvidia.com/blog/nvidia-vera-rubin-post-training-intelligence-per-dollar/) also connects rollout generation with training updates. Study rollout/trainer resource balance as an extension; prefix reuse across weight updates requires invalidation because the computation has changed.

The companion [AgentOS course](../../llm-agentos-course/) teaches harness policy and memory orchestration. This topic measures the engine/cache behavior under those policies, preserving the distinction between agent memory and serving state.
