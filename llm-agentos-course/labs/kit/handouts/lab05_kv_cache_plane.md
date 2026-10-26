# Lab 5: the KV-cache plane

**Lecture:** [the KV-cache plane](../../../slides/03_kv_cache_deck.html), Module 3 · **Week:** 5 ·
**Autograder:** `make grade LAB=5` · **Budget:** Parts A and B run offline. Part C needs one GPU with 24 GB or
more for a 7–8B model, and Part D also needs the LMCache setup from
[`../../lab2_aios_kv_cache/`](../../lab2_aios_kv_cache/).

Agents are a prefill problem. Every turn resends the tools, the system prompt and the whole history, and then
adds a little. A prefix cache serves the repeated part only when the new request starts with exactly the same
tokens as an earlier one. vLLM checks this by hashing the prompt in 16-token blocks, and each block's hash is
chained to the one before it. One changed token therefore invalidates every block after it. In this lab you:

- rebuild that check on the harness side
- break the cache on purpose, offline
- do it again on a real vLLM server, and then in Hermes or OpenClaw

## Part A: a prefix-stability checker (graded)

Implement the three TODOs in `nanoos/prompt.py`, following their docstrings:

1. **`block_hashes`**: vLLM-style chained SHA-256 hashes over full blocks. An optional salt plays the role of
   the per-tenant `cache_salt`.
2. **`PrefixCache.access`**: a simulated engine cache with a fixed number of blocks.
   - Count hits from block 0 up to the first miss.
   - Use LRU eviction, and evict a request's tail before its head.
3. **`diagnose`**: compare a session's previous and current prompt, find the first changed block, and name the
   likely cause. The causes are:
   - `tools`: the tool list changed.
   - `timestamp`: a timestamp in the system prompt changed.
   - `memory`: an edit inside the memory block that Lab 4 injects.
   - `system`: any other change to the system prompt.
   - `history`: an earlier message was rewritten, as compaction, tool-result clearing or unstable
     serialization do.

Rendering is provided (`segments`, `render`). It puts the tools first, then the system prompt, then the
messages, which is the order that providers cache. A stand-in tokenizer (`cache_tokens`) is also provided,
along with `CacheAuditModel`, which wraps any model and records hits and causes for every call.
`make grade LAB=5` runs 14 tests.

## Part B: break the cache on purpose, offline (report)

`tasks/cache_session.py` scripts a 20-turn session. Its prompt grows to about 10k tokens. The deck's variants
each make one change:

- `timestamp`: a timestamp in the system prompt.
- `tool_at_10`: a tool added at turn 10.
- `memory_at_10`: a memory edit at turn 10.
- `compaction_15`: compaction at turn 15.
- `masked_tool_10`: Manus's alternative to adding a tool. The tool is listed from the start and only allowed
  at turn 10.

```bash
python cache_experiment.py                         # every variant, one session, unlimited cache
python cache_experiment.py --variants baseline --sessions 8 --cache-blocks 3000    # sessions compete for KV
```

Report:

- Hit rate per turn for each variant, as a plot or table from `--csv`. Also report prefill tokens per session
  and the cost relative to no cache. The defaults use the deck's multipliers: reads at 0.1× and writes at
  1.25×. Why does the `timestamp` variant cost *more* than no caching at all?
- How many blocks of KV each session needs by turn 20. Then sweep `--sessions` at a fixed `--cache-blocks`
  and find the most concurrent sessions before the hit rate falls below 80%.
- Turn your `--cache-blocks` into GiB with the deck's formula: bytes per token = 2 × layers × KV heads ×
  head dim × bytes per element. Take the values from your model's `config.json`. How many sessions would fit
  in 16 GiB of KV at FP16, and how many at FP8?

## Part C: the same session on vLLM (report)

Serve a model with prefix caching. Ask the server to report cached tokens in `usage`:

```bash
vllm serve Qwen/Qwen3-8B --enable-prefix-caching --enable-prompt-tokens-details
export NANOOS_BASE_URL=http://localhost:8000/v1 NANOOS_API_KEY=local NANOOS_MODEL=Qwen/Qwen3-8B
python cache_experiment.py --model api --csv runs/lab5/vllm.csv
```

Each request is sent with `max_tokens=1`, so its time is mostly prefill, close to the time to first token.
Every run starts its prompts with a fresh tag, so an earlier run cannot warm the cache. Then repeat the
baseline twice:

- with `--kv-cache-dtype fp8`
- with LMCache CPU offload, using [`run_vllm_lmcache.sh`](../../lab2_aios_kv_cache/run_vllm_lmcache.sh). Add
  `--enable-prompt-tokens-details` to it.

To force eviction, shrink the KV cache with `--gpu-memory-utilization` or `--num-gpu-blocks-override`, then
sweep `--sessions` again. Check `vllm serve --help` for the flags your version has.

Report:

- The simulated hit rate next to the server's (`api_cached_tokens`), and the latency per turn for each
  variant. The server's chat template renders tools and messages differently from `render`. Where do the
  simulated and real breaks agree, and where not?
- The most concurrent sessions before eviction with FP16 KV, with FP8 KV, and with LMCache offload. Read
  `vllm:kv_cache_usage_perc` from `/metrics` while the sweep runs.

## Part D: a production harness on the same server (report)

Point Hermes or OpenClaw at the same server, using its custom OpenAI-compatible provider with base URL
`http://localhost:8000/v1`. Run a 20-turn session of your own, and read vLLM's prefix-cache counters before
and after:

```bash
python cache_experiment.py --scrape http://localhost:8000/metrics
```

Report the harness's hit rate. Find one thing in its prompt that breaks the cache, or show that it does not
break, and say which item of the deck's cache-hygiene checklist explains it. You can save the requests the
harness sends with a logging proxy, then feed them to `diagnose`.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=5`) | 40% |
| Report: offline variants, the cost question, the KV sizing | 20% |
| Report: vLLM results (simulated vs. real, FP8, LMCache, concurrency) | 25% |
| Report: the production harness | 10% |
| Code quality of `prompt.py` | 5% |

## Stretch

- **A real tokenizer.** Replace `cache_tokens` with your model's tokenizer and chat template
  (`transformers`' `apply_chat_template`). Check whether the simulated breaks now match the server's exactly.
- **nano-vllm.** In `GeeeekExplorer/nano-vllm`, add a prefix-cache hit counter per request, or a TTL that keeps
  a session's blocks during a tool call (the Continuum idea from Lab 2), and measure it.
- **Cache-stable assembly.** Write `assemble(...)` in `prompt.py`, which builds the prompt from tools, system,
  frozen memory and history so that none of the variants above breaks the prefix. Show it with
  `CacheAuditModel` on a Lab 3 long-tier run.
- **Prefill/decode split.** Try the NIXL setup in
  [`vllm_lmcache_nixl_config.py`](../../lab2_aios_kv_cache/vllm_lmcache_nixl_config.py) on two GPUs.
