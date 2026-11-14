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

<a id="matched-memory-evaluation"></a>
## Optional extension: compare memory conditions fairly

Companion to long-term-memory slides 17/19 and observability slide 19; implements audit item **AGT-2's teaching/protocol update**. The released `memory_eval.py` compares **read paths on one extracted store** (`none`, `auto`, `tool`, `both`). It does not implement full-history or verbatim-history baselines. The protocol below requires those adapters and real answer-model runs before it yields model-quality results. No new launcher or benchmark result is supplied.

### 1. Declare the comparison

Freeze history/question IDs, user scope, answer-model/tokenizer revision, sampling/retry settings, judge/rubric, input cap, output cap, trial count and run order. Keep the available input ceiling the same; actual consumed tokens are an outcome to report, not a quantity to force equal. If history exceeds the cap, mark full-history **ineligible**. A larger-window diagnostic belongs in a separately labeled budget group; do not silently truncate or pool it with the equal-cap comparison.

| Condition | What reaches the answering model | What this comparison can test |
|---|---|---|
| No memory | Question, without prior user history | Whether the task needs retained history |
| Full history | Complete eligible history | Whether the model can use the information when supplied; not a guaranteed quality ceiling |
| Verbatim retrieval | Retrieved original history chunks | Retrieval without learned fact extraction |
| Extracted memory | Retrieved facts written by an extractor | The combined write/retrieve design, with extraction cost |

For verbatim versus extracted memory, match embedding identity, retrieval settings and eligible questions where feasible. Different object lengths can prevent identical chunking; record this and any other mismatch. Comparing whole pipelines that differ in several components does not isolate extraction as the cause. Counterbalance repeated-run order or explain how warmup/cache effects are controlled. Start each independent condition from a clean store built only from that condition's allowed history. Keep answers and grading labels out of its write path.

### 2. Keep quality and accounting together

Retain per-question/trial outcomes, question type, input/output tokens, errors, truncation and eligibility. Use the same rubric and a frozen judge; blind condition names and inspect a human-labeled sample for disagreements. A common judge does not guarantee all biases cancel. Pair conditions by question ID and trial policy; report both paired coverage and missing-case reasons. Disclose judge cost separately.

| Condition / task type | Eligible n / planned n | Quality and uncertainty | Write calls / tokens / time / cost | Read + answer tokens / time / cost | Total cost / failures |
|---|---|---|---|---|---|
| No memory | Fill after run | Not measured | Measure or justify zero history-write work | Not measured | Not measured |
| Full history | Record fit / exclusion | Not measured | Include any preprocessing actually performed | Not measured | Not measured |
| Verbatim retrieval | Fill after run | Not measured | Include indexing/embedding, even with no extractor LLM | Not measured | Not measured |
| Extracted memory | Fill after run | Not measured | Include extraction, consolidation, indexing and embedding | Not measured | Not measured |

Report cold ingest cost and steady read cost separately. If amortizing writes over Q future questions, state Q and show `total = write_cost + sum(read_answer_cost)` alongside cost per answered question. Do not mix currency with tokens or treat CPU time as billed API cost. The current demo scripts answers, so its `success` column is **not** evidence of model capability; its retrieval/scope observations have a different scope. Current kit metrics do not automatically supply every time/cost field above: instrument adapters or mark fields unavailable.

### 3. Prediction checkpoint: mismatched subsets

Constructed teaching counts, **not measured results**:

- A answers 80 of 100 questions correctly. B answers 18 of 20 on a smaller subset. Can 80% versus 90% rank their general quality? **No:** question sets differ.
- On the same 20 questions, A gets 16 correct and B gets 18. The paired-subset marginal difference is **+10 percentage points for B**. It says nothing about B on the other 80 questions.
- Per-question outcome pairs are still needed for a paired uncertainty/significance calculation; marginal totals alone omit the disagreement structure. Do not invent confidence intervals or infer a universal winner.
- No time or monetary cost was given, so this example cannot establish efficiency.

Submit your prediction, the declared controls, per-question records, a matched-subset table, missing-case reasons and one failed example. A source-only submission leaves all measurements marked not-run. An implementation submission must distinguish implemented adapters from proposed ones and show the actual environment. The existing graded Lab 4 tasks and reference behavior remain as above.

Primary reading: Kuan Wang, [MemDelta, arXiv:2606.29914v1](https://arxiv.org/html/2606.29914v1), §3 controlled comparisons, §4.4 costs and limitations. The paper's partial coverage and imperfect embedding matching limit general claims; this worksheet is an original teaching protocol, not a reproduction of its tables. LoCoMo and LongMemEval remain benchmark readings in the course bibliography; their historical results are not universal performance guarantees.

中文检查：先固定问题、历史、模型、评分和预算，再比较四种条件。不同问题集合的百分比不能直接排名；共享 20 题的 +10 个百分点仅适用于该子集。真实实验需记录写入与读取总成本，缺失结果不能填成通过。

## Measured standalone pilot — 2026-11-14 UTC

The [real-model pilot](../../../references/measured_memory_pilot_2026_11.md) now supplies all four standalone conditions and measured write/answer records for eight matched synthetic questions. It does not use the student MemoryStore TODOs or the scripted answer model. The frozen primary strict scores are all 0/8; inspect the raw text, formatting false negatives and factual errors before interpreting them. Use the separate `memory_pilot.py` write/answer phases with the pinned protocol; the broader semantic/human/large-cohort benchmark remains open.
