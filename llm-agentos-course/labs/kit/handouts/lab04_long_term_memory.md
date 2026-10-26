# Lab 4: long-term memory

**Lecture:** [long-term memory](../../../slides/02_long_term_memory_deck.html), Module 2 · **Week:** 4 ·
**Autograder:** `make grade LAB=4` · **Budget:** API only. About 40 to 80 short calls per read path, plus
one extractor call per past message with `--extractor model`. Part C needs a Letta server, local or Letta Cloud.

The model is stateless. Anything it "remembers" was written to a store and later put back into the prompt by
the harness. In this lab you build that small data system end to end:

- a write path that extracts facts and consolidates them
- a hybrid index
- the deck's two read paths

Then you run the deck's exercise: the same store with two read paths, on 20 questions that need memory and 20
that do not. The lab follows DeepLearning.AI's *LLMs as Operating Systems* course, which builds memory from
scratch first and then in Letta (see [`LAB_PLAN.md`](../../LAB_PLAN.md), section 2).

## Part A: build the memory system (graded)

Implement the four TODOs in `nanoos/memory.py`, following their docstrings. BM25, the embedders, persistence,
the extractors and read path B (`MemoryTools`) are provided.

1. **`MemoryStore.apply`: consolidation.** Memories must change, not only pile up. Apply Mem0-style
   operations (ADD, UPDATE, DELETE, NOOP):
   - Duplicates are dropped.
   - A memory can hold a profile slot (`key`), and each user has at most one active memory per slot.
   - UPDATE *supersedes* the old memory and keeps it as history, the bi-temporal idea from Zep.
   - DELETE is the right to forget: every version is removed, so not even the saved file keeps the text.
2. **`MemoryStore.ingest`: the write path and its trust gate.** Extract from what the user said. An
   untrusted tool result never reaches the extractor, because a web page that says "remember that..." is
   memory poisoning. This is the deck's poisoning row, and Lab 6 builds on it.
3. **`MemoryStore.search`: hybrid retrieval.** BM25 and vector similarity, fused with Reciprocal Rank
   Fusion. Search only the user's own active memories, since a scope check is how cross-user leaks are
   prevented.
4. **`AutoRecall.prepare`: read path A.** Before each model call, search with the latest user message and add
   the hits to the system prompt. Read path B is `MemoryTools`, which gives the model a `search_memory` tool.

`make grade LAB=4` runs 17 tests. The last one runs the full exercise offline and checks two things: that no
forgotten, poisoned or other-user fact ever reaches the model, and that every needed fact is surfaced.

## Part B: same memory, two read paths (report)

The fixtures in `tasks/memory.py` have four users, each with three past sessions. The sessions include a move,
a changed deadline, a request to forget, and two poisoned web pages. There are 40 questions, of six kinds:
fact, update, forget, poison, other user, and absent (never said). Start offline. `--model demo` scripts the
answers, but the retrieval columns are real:

```bash
python memory_eval.py --read auto --model demo     # one question misses its fact at k=3: find out why
```

Then run your model with each read path:

```bash
for r in none auto tool both; do
  python memory_eval.py --read $r --model api --trials 3 --loop nanoos --out runs/lab4-$r
done
```

Then rerun the best path with `--extractor model`, so your model does the extraction instead of the gold
script. If you have an embedding endpoint, also add `--embedder api` (set `NANOOS_EMBED_MODEL`).

Report, per read path:

- Accuracy on the 20 memory questions and on the 20 others, and accuracy per kind (`by_kind` in
  `summary.json`).
- Input tokens per question, split the same way. What does path A cost on questions that need no memory?
- **How often the model calls `search_memory`** on each set (`tool_call_rate`). The deck's warning is that
  "the model may not realize it should search". Did it, and did it search when it did not need to?
- Recall (did the needed fact reach the model), leaks and stale answers.
- With `--extractor model`: what did your model's extraction get wrong compared with the gold operations?
  Diff `runs/.../memory.json` against the gold run's. Look in particular at the UPDATE, the forget and the
  poisoned pages.

Then answer: path A changes the system prompt whenever the retrieved memories change. What does that do to a
prompt-prefix cache? Hermes freezes its memory snapshot for the session to avoid this (see the deck's
hybrid-pattern slide), while OpenClaw refreshes it each turn. Which would you choose, and why? Lab 5 measures
the cache.

## Part C: the same store in Letta (report)

Letta is the purest form of path B. Core memory blocks are always in context, and archival memory is reached
only through the agent's own tools. Export the store and load it into a Letta agent per user:

```bash
python memory_eval.py --read none --export-letta runs/letta.json
```

Each user in the export has a `human_block` (the keyed profile facts) and `passages` (all active memories).
With the Letta Python SDK, a minimal version is: create the agent with `memory_blocks=[{"label": "human",
"value": human_block}]`, insert each passage with `client.agents.passages.insert(agent_id=..., content=...)`, then
ask the 40 questions through `client.agents.messages.create`. Check the current Letta docs for exact
signatures, since the SDK changes often. Run it twice: once with everything in the core block, and once with
only passages, so the agent must call `archival_memory_search`. Compare both runs with your path A on
accuracy and tokens. You can score the answers with the `check` functions in `tasks/memory.py`.

Hand in one page from [`report_template.md`](../report_template.md).

## Grading

| Part | Weight |
|------|-------:|
| Autograder (`make grade LAB=4`) | 50% |
| Report: the read-path table, tool-call rates, the model-extractor diff, the cache question | 30% |
| Report: the Letta comparison | 10% |
| Code quality of `memory.py` | 10% |

## Stretch

- **A real index.** Replace the in-memory index with LanceDB or SQLite (FTS5 plus `sqlite-vec`) behind the
  same `search` method, and keep the tests passing.
- **Write before compaction.** Call `ingest` from your Lab 3 `ThresholdCompaction` on the chunk it is about to
  summarize, as Hermes's `on_pre_compress` and OpenClaw's pre-compaction flush do. Then run the long tier.
- **Scoring beyond relevance.** Add recency decay and an importance score, as in Generative Agents, and show
  one question where the ranking changes.
- **A hot-path write tool.** Give the model a `remember(text)` tool next to `search_memory`, and compare what
  it chooses to store with the background extractor.
