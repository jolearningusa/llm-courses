# Session handoff, pending operations and restart controls

Executed 2026-11-09 for AGT-4. [Provided fixture](../labs/kit/session_handoff_eval.py), [tests](../labs/kit/tests/test_session_handoff.py), [recorded CPU cases](session_handoff_cpu_2026_11.json). Python 3.12.14 / Darwin, standard library only. Every case starts a **new Python interpreter** that reads saved context and fake-tool receipts from temporary files. No LLM, external service or real tool action runs.

## Sources and what they establish

[Anthropic's long-running harness study](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents) describes progress artifacts and startup checks that help new sessions recover the state of work. It motivates explicit handoff state, not a guarantee of unlimited autonomy or our fixture's outcomes.

[Amazon Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) discusses caller request identifiers, equivalent retry responses and rejecting changed intent under the same identifier. It also requires suitable server-side atomic/durable behavior. Our local JSON stub models a small part of that contract; it is not an AWS implementation, distributed transaction or production exactly-once guarantee.

## One task, three declared interruption boundaries

The fixed task is to prepare and publish one local report. The `prepare` step is already complete; `publish` is pending with its stable key and payload. A completed fake-tool effect means only a recorded receipt in a temporary JSON ledger, not an actual report delivery.

| Cut | Saved client state | Fake-tool state before restart | Correct recovery in this fixture |
|---|---|---|---|
| Before tool | Pending operation with key/payload; prepare complete | No effect | Apply once, then mark publish complete |
| After effect, before receipt saved in handoff | Pending still present | One effect/receipt already recorded | Retry the same key/payload and obtain the saved receipt, without a new effect |
| After receipt saved | Prepare/publish complete; pending absent | One effect/receipt | Preserve completed work; no new tool attempt |

This simulates a session boundary **after the specified writes finish**. It does not kill a process in the middle of a write, verify `fsync`, test torn files or prove crash atomicity. The new worker has no access to parent Python objects; its arguments are only context/receipt file paths. The fixed task ID is part of the fresh request in both arms. Goal, plan, completed work and pending operation must come from retained context, not be filled in by the scorer.

## Equal declared budget, with a useful negative control

Both inputs are compact JSON padded with spaces to exactly **512 ASCII bytes**:

- **Handoff:** a structured state snapshot containing task ID, goal, step plan, completed steps and pending operation including stable key/payload.
- **Naive tail:** the largest whole-event suffix of the transcript that fits the same cap. The reader reconstructs state by applying retained event fields. Sixteen irrelevant log events evict the required earlier state.

The same worker validates and reads both formats. A short-tail control omits the irrelevant log events; its required records fit and all three cuts recover. This rules out a parser that simply rejects every transcript. The recorded cases include actual serialized-content size, padding, input context, worker exit code, recovered state, attempt count and total fake effects.

A byte cap is **not** a real tokenizer budget, token-cost match or model-quality comparison. Padding establishes equal allocated input capacity, not equal semantic content or measured inference work. A production harness may pin goals and tool state outside the transcript; this deliberately naive tail is not its implementation.

## Observed outcomes

| Cut | Handoff status | Post-restart calls / total fake effects | Noisy-tail status | Noisy-tail total fake effects |
|---|---|---|---|---|
| Before tool | complete | 1 / 1 | blocked | 0 |
| After effect, before saved client receipt | complete; saved receipt replayed | 1 / 1 | blocked | 1 |
| After saved client receipt | complete | 0 / 1 | blocked | 1 |

Handoff preserves the goal and completed `prepare` step and resolves pending state at all three cuts. Noisy tail fails closed when mandatory state is absent: it does not guess a goal or retry a tool. The three short-tail cases all complete with one total fake effect. These are nine fixed cases of one task, not independent task samples or an LLM reliability rate.

Additional unit controls demonstrate that:

- A new retry key after an already recorded effect produces **two** fake effects, even with the same payload.
- The same key/payload returns the existing receipt; changing the payload under that key is rejected without changing the saved ledger.
- Missing operation keys, wrong task IDs, missing checkpoints, invalid/oversize JSON and inconsistent progress do not become tool actions or successful completion.
- Having no pending operation alone is insufficient to declare every planned step complete.

Schema validation is not authenticity or truth verification. The stub is single-caller, retains keys indefinitely and assumes intact serial JSON files. TTLs, concurrent retries, actual external side effects, authenticated state, process/power failure during writes and cross-service commit/reconciliation remain unqualified.

## Reproduce

From `llm-agentos-course/labs/kit` with Python 3.10+:

```sh
python3 session_handoff_eval.py > session-handoff.json
python3 -m unittest tests.test_session_handoff tests.test_memory_repair tests.test_kit -v
```

The captured fixture contains six main cases plus three short-tail cases, each with a fresh subprocess. Compare cases and the fixture-source SHA-256; environment metadata may differ on another host. Twelve new tests plus ten prior memory-repair tests and 36 provided-kit tests passed (58 total). The existing kit's two loopback HTTP fixtures contact no external service; the handoff fixture itself uses no network. Student `nanoos` TODOs and original Lab 3 compaction engines are unchanged.

## Lab extension and limits

Retain the original 40+ turn/model comparison in Lab 3 as a separate exercise. For an actual model/harness recovery study, fix task distribution and tokenizer budget, declare durable-state inputs and access, include fresh and stale handoffs, and instrument both preserved task utility and repeated/omitted operations. Verify the real tool's retry/reconciliation, retention-window and payload-identity contracts before interpreting an acknowledgement-loss experiment. Distinguish intentional stop/resume from process crashes and power-loss durability; record unsupported/not-run cases explicitly.

This completes the bounded AGT-4 deterministic handoff fixture and teaching material. It does not qualify a production context engine, actual model compaction, restart reliability under arbitrary failures or real-service exactly-once execution. AGT-2's measured memory benchmark still remains pending beyond PR #120's protocol/teaching work.
