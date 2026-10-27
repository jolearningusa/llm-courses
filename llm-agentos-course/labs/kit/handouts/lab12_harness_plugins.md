# Lab 12: porting to real harness plugin APIs

**Lecture:** [open-source plugin APIs](../../../slides/05_opensource_plugin_apis_slides.html) and the
[harness landscape](../../../slides/05_harness_landscape_deck.html), Module 5 · **Week:** 12 ·
**Autograder:** `make grade LAB=12` · **Budget:** Part A and the offline runs are free. Part B needs 3 long tasks
× 3 trials × 4 configurations with your model, about 36 runs of 40+ steps. Part C needs Hermes Agent, OpenClaw
and a Letta server.

Labs 3 and 4 built a context engine and a memory store behind nanoos's own small interfaces. This lab ships
them as plugins for production harnesses, without changing them. It reworks the course's first plugin lab
([`../../lab1_memory_plugins/`](../../lab1_memory_plugins/), which keeps its folder name):

- **12A** the Hermes plugin now wraps your engines;
- **12B** the OpenClaw plugin is now a thin TypeScript shell around your Python engine;
- **12C** the Letta exercise is unchanged.

The lesson is in the mismatch. A nanoos engine returns a **view** of an append-only history on every call.
Hermes offers it two places:

- `compress()` **rewrites** the stored transcript once the prompt crosses a threshold;
- `select_context()` changes **one request** and leaves the transcript whole.

OpenClaw's `assemble()` is a view under a token budget. An engine that sets `ownsCompaction` must shrink its own
store in `compact()`, because OpenClaw then turns off its built-in compactor.

## Part A: three adapters (graded)

Implement the TODOs in [`nanoos/ports.py`](../nanoos/ports.py). [`nanoos/hosts.py`](../nanoos/hosts.py) calls
them the way each harness does, so you can watch the hook order offline.

1. **`HermesContextEngine`** (Hermes's `ContextEngine`).
   - `should_compress`: true only in `compress` mode, at the threshold (75% of the window by default).
   - `compress`: a **fresh** engine each time, built for half the threshold, over Hermes's snapshot. It must
     be fresh because the transcript it sees next is the rewritten one, so a long-lived engine's positions
     would be wrong. Count a compression only when your engine compacted.
   - `select_context`: in `select` mode, one engine for the session, since the transcript only grows. Return
     `None` when nothing changed, so Hermes keeps its own list.
2. **`HermesMemoryProvider`** (Hermes's `MemoryProvider`), around your Lab 4 `MemoryStore`.
   - `initialize`: open the store under `hermes_home`, scoped to Hermes's `user_id`.
   - `prefetch`: return the warmed result for this query if `queue_prefetch` made one, else recall now.
   - `sync_turn`: hand the write to a background thread. Hermes requires that it never blocks the loop.
   - `on_pre_compress`: write now, before the engine drops the messages. Your Lab 4 trust gate still decides
     which tool results count.
   - `on_session_end`: wait for pending writes, then save.
3. **`OpenClawEngine`** (OpenClaw's context-engine contract).
   - `assemble`: your engine's view of the session's log, then cut to `tokenBudget`. After a compaction, add
     a `contextProjection` with the new epoch.
   - `compact`: rewrite the log with a fresh engine at half the budget; a `/compact` forces it. Without
     `ownsCompaction`, do nothing.

The tests use stand-in engines and a stand-in store, so this lab's grade does not depend on Labs 3 and 4.
`make grade LAB=12` runs 11 tests. They cover:
- the hook order in the Hermes host, with `on_pre_compress` right before `compress`;
- that `sync_turn` returns while a slow extractor is still running;
- the stdio bridge;
- that the OpenClaw host keeps replying, on its fallback, when an engine raises.

The plugin files load your adapters into the real harnesses:

- [`hermes_custom_memory.py`](../../lab1_memory_plugins/hermes_custom_memory.py) registers both adapters.
  `python hermes_custom_memory.py` runs a two-session memory demo offline.
- [`openclaw_context_engine.ts`](../../lab1_memory_plugins/openclaw_context_engine.ts) starts
  `python3 -m nanoos.ports` and forwards each call as one JSON line.

## Part B: views against rewrites (report)

Run the long tier through each port. The scripted runs compact because the window and the budget are small:

```bash
python ports_eval.py --host hermes --engine none --out runs/lab12-none
python ports_eval.py --host hermes --engine threshold --mode compress --out runs/lab12-h-compress
python ports_eval.py --host hermes --engine threshold --mode select --out runs/lab12-h-select
python ports_eval.py --host hermes --engine clearing --mode select --out runs/lab12-h-clearing
python ports_eval.py --host openclaw --engine threshold --bridge --out runs/lab12-oc
```

The scripted model's answers are fixed, so offline success is always 100%. Offline, look at:
- the prompt tokens;
- the peak prompt;
- the number of rewrites;
- the **cacheable share**: the prompt tokens that repeat the previous request's leading messages, a stand-in
  for what a prefix cache could reuse (Lab 5).

On the scripted runs:
- threshold compaction cuts the prompt tokens by more than half in both modes;
- the rewrite spends fewer tokens than the view, because later turns start from the shorter transcript;
- clearing spends the fewest tokens, but only about 40% of each prompt repeats the one before.

Explain all three from the requests in `hosts.HostResult`.

Then run the threshold engine in both Hermes modes, clearing, and the OpenClaw bridge with your model
(`--model api --trials 3`; the summaries then come from your model). Report:

- success, pass^3 and prompt tokens per configuration;
- the cacheable share against the `cached_input_tokens` your provider reports;
- where a rewrite lost something a view kept: find a run where the instruction from the start of a task was
  followed in `select` mode and lost in `compress` mode, or show that none was lost;
- the bridge's cost: the time per `assemble` over stdio, against the time per model call.

## Part C: the real harnesses (report)

**Hermes Agent (12A).** Install the plugin as its header says, and select `nanoos-memory` and `nanoos-context`.

1. Write each long task to a folder with `python export_task.py l01_sum_parts /tmp/l01 --tier long`. Run it 3
   times with your engine and 3 times with Hermes's built-in compressor (no `context.engine`).
2. Compare the prompt tokens, compressions (Hermes's status line reads `compression_count`) and answers.
3. Memory: tell Hermes a fact in one session and ask for it in the next.
4. Answer from the real harness:
   - Where did the prefetched text appear in the prompt? The kit's host adds it to the user message; Hermes may
     not.
   - Does your Hermes version have `agent.memory_provider.spawn_context_thread`? If so, use it in `spawn()` and
     say why a bare thread could write to the wrong profile.

**OpenClaw (12B).** Install the shell as its header says, with `kitDir` pointing at your kit.

1. Run the same tasks with `plugins.slots.contextEngine` set to `nanoos-engine`, then to `legacy`.
2. Make your `assemble` raise on the tenth call and show what OpenClaw does: the engine should be quarantined
   for the rest of the gateway process.
3. Check the engine with `openclaw doctor`.

**Letta (12C).** Run [`letta_shared_blocks.py`](../../lab1_memory_plugins/letta_shared_blocks.py) and show the
supervisor's write appearing in the worker's context on its next turn. Compare the three memory models in a few
sentences:
- a block that is always in context and shared by agents;
- your per-user store, recalled on demand;
- OpenClaw's memory slot.

For each one, say who decides what is remembered and who can see it.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=12`) | 50% |
| Report: views against rewrites with your model, the bridge's cost | 20% |
| Report: the three harnesses, including the quarantine and the memory session | 20% |
| Code quality of `ports.py` | 10% |

## Stretch

- **Focus.** Hermes passes `focus_topic` from `/compress <topic>`. Pass it to your summarizer and show a
  compaction that keeps what the topic names.
- **Sub-agents.** Implement OpenClaw's `prepareSubagentSpawn` so a Lab 9 child starts from the parent's
  summary instead of an empty log, and measure the child's tokens.
- **Survive a restart.** Persist the OpenClaw session logs in `afterTurn` and reload them, so a gateway restart
  does not lose the compacted history.
- **One process, many sessions.** Run the bridge under load (10 sessions in parallel) and find what in
  `OpenClawEngine` needs a lock.
