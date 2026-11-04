# L7 — State ownership and inference-cache mechanisms

**Released:** standard-library logical full-block cache simulation. **Source-inspected:** vLLM, SGLang and llama.cpp documentation. **Not executed:** engine inference/two-engine benchmarks. Estimated 5–8 hours, student pilot pending.

From `labs/kit/` run `python3 cache_lab.py > cache-model.json` and the unit suite. The reference stores only identity/token keys and counts, not tensors; it implements no attention, sampling, partial-block state, async transfer or real allocator. The simulator uses simple full-prefix tuple keys and idle LRU; these are teaching choices, not literal vLLM/SGLang code.

## A. Derive and test ownership

Independently draw key/refcount transitions for two shared eight-token requests with block size four. Cancel one, verify the other owns two pinned blocks, release it, then evict under pressure. Test a nine-token warm prompt (eight eligible tokens), early/late mismatch, model/domain changes and a failed admission that must not leak references. The supplied complete model is a reference; student task: add an explicit finite physical-block ID/free-list representation and preserve these invariants.

Correctness gates: never reuse an incompatible prefix; never evict pinned state; release only the requesting owner; recover cleanly from failed admission; distinguish resident blocks, pinned blocks and total references. The simulation deliberately accounts only for contiguous full-block hits. It is not a latency predictor.

## B. Compare two actual engines, or complete the source worksheet

Use vLLM and llama.cpp where available, or substitute a documented SGLang configuration. Create separate environments and pin commits/releases, model/tokenizer revisions, backend, dtype/layout, context capacity, slots/concurrency and cache flags. No shared portable session/KV format is assumed. The authoring host did not run these engines; the following is an experiment protocol, not a validated launcher.

| Case | Fixed token workload | Record for each engine |
|---|---|---|
| Cold | New process or documented tensor-cache reset | Initialization separately; evaluated prompt tokens; TTFT |
| Shared prefix | Identical long prefix, divergent final question | Matched/avoided tokens; actual counter source |
| Early mismatch | Change first eligible block | Prefix hit loss; evaluated tokens |
| Late mismatch | Change late block | Surviving match length |
| Pressure | More active conversations than slots/capacity | Live/idle bytes, eviction/recompute, rejection |
| Cancel/share | Cancel one owner of a shared prefix | Surviving correctness and reclaimed ownership |
| Invalidate | Change model/adapter/domain | Documented key/reset boundary; no incompatible hit |

For llama.cpp inspect `cache_prompt`, slot behavior, RAM prompt cache and save/restore separately. For vLLM inspect full-block identity/refcounts and cache metrics in the chosen version. For SGLang inspect radix/hierarchy configuration. Document numerical/sampling behavior; identical sampled strings alone are not a valid correctness gate. Report matched token IDs and generation settings before timing.

Without an engine, fill the table with cited mechanism/metric locations and explicit “not measured” cells. Explain expected outcomes as hypotheses. Never replace missing counters with invented hit counts or present simulator output as an engine result.

## C. Response-cache distinction

Read [cache-layer notes](../../../notes/agentic_caching.md). Explain how GPTCache-style semantic response reuse differs from exact prefix tensor reuse. Construct an example where stale tool output makes a semantically close cached answer invalid while compatible prefix KV remains reusable. Submit raw simulation JSON, two-engine/source worksheet and [report](../../report_template.md).

<a id="cache-teaching-checkpoint"></a>
## Cache teaching checkpoint — deck 16

Allow 10–15 minutes (estimate). Predict before execution: with four-token blocks, A holds tokens 0–7 and B requests tokens 0–8. How much does B match? Does releasing A evict the shared state?

From `labs/kit/`:

```sh
python3 - <<'PY'
from cache_lab import Identity, PrefixCache
cache = PrefixCache(capacity=4, block_size=4)
identity = Identity('checkpoint')
cache.acquire('A', list(range(8)), identity)
print('B', cache.acquire('B', list(range(9)), identity))
print('both', cache.snapshot())
cache.release('A')
print('after_A', cache.snapshot())
cache.release('B')
print('after_B', cache.snapshot())
PY
```

Expected B fields: `matched_tokens=8`, `recomputed_tokens=1`, `full_blocks_owned=2`. Expected snapshots:

| Boundary | Resident blocks | Pinned blocks | References | Live requests |
|---|---:|---:|---:|---:|
| Both own the prefix | 2 | 2 | 4 | 2 |
| A released | 2 | 2 | 2 | 1 |
| B released | 2 | 0 | 0 | 0 |

The partial ninth token is counted as recomputed but is not persisted as a full block. Released idle blocks remain resident until eviction. Extend the exercise with a first-token mismatch and, separately, a changed sharing domain: start each case from a fresh cache populated only by A, and predict zero matched tokens. Later equal token IDs cannot repair a different prefix history. Read the [vLLM identity/refcount design](https://docs.vllm.ai/en/latest/design/prefix_caching/) as an implementation comparison, not proof that this tuple-key simulator implements that engine.

Submit the four counter states, changed-case predictions and one reason simulated avoided tokens cannot predict measured TTFT. Then complete B's engine/source worksheet, explicitly marking unavailable measurements.

中文提示：B 命中 8 个 token、重算 1 个；A 释放后，共享块仍由 B 固定。空闲驻留不等于已淘汰。模拟器没有真实 KV 张量，也不预测 TTFT。
