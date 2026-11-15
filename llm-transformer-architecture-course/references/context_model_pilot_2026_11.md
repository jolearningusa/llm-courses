# ARC-3 fixed-checkpoint context pilot — 2026-11-15 UTC

**90 real model generations completed; strict score 0/81 on the grid and 0/9 on evidence-only controls.** All planned inputs fit. Because the short controls also fail, this pilot cannot identify a context-length or position effect, establish an effective window, or validate PI/YaRN. The earlier 81/81 grammar-oracle agreement remains fixture evidence, not model accuracy.

## Primary reading and frozen design

[RULER v1](https://arxiv.org/html/2404.06654v1), §3, §5 and Appendices B/D, motivates separating retrieval, tracing and aggregation, checking distractors and documenting chat/task templates. Its benchmark uses different tasks, larger lengths and answer-prefix/demonstration choices. We reuse our existing course-authored lookup/two-link/count fixtures, not RULER data, implementation, scores or its effective-window threshold.

The pinned [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) documents CPU inference and limitations in accuracy/instruction following. The [pinned config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/config.json) declares8192 positions, RoPE theta100000 and no scaling. This configured bound is not a measured quality guarantee.

Protocol, runner and tokenized [plan](../labs/L3_positions/context_model_artifacts/plan.json) were committed at **309b8ae79461938161d6a44159054912a6615d69 before inference**. Base main was `1d1e06d`. No checkpoint, prompts, length bins, labels or scorer were tuned after viewing results; one trial, greedy decoding, no retries and no dropped cases. The plan contains90 unique IDs: existing81 cases (32/64/128 records × early/middle/late × lookup/chain/count ×3 variants, fixture seed7), plus9 controls retaining only each task/variant's evidence records. Controls run first; the grid has a fixed seed20261115 shuffle. This order is exposed, not a timing-replication claim.

Labels/IDs/target metadata stay outside the message builder, which accepts records/question/system text only. The factual value remains present in or derivable from evidence as intended. Questions, answers and filler sequences are paired across placements; the same task/variant recurs across lengths. Evidence is a **contiguous bundle**, not dispersed multi-evidence reasoning. These are only nine underlying task/variant instances: each appears in nine grid conditions plus one short control, not90 independent sampled tasks or a natural-language corpus.

## Execution contract and token locations

Reused local checkpoint/tokenizer revision **12fd25f77366fa6b3b4b768ec3050bf629380bac**, all nine file SHA-256 identities checked against the [prior download ledger](../../llm-agentos-course/references/memory_pilot_artifacts_2026_11.json). Model weights SHA matches the previously retrieved hub LFS identity; no new weight download, training or package installation occurred. `local_files_only=True`, offline hub/Transformers settings and `trust_remote_code=False`; no credential, endpoint or paid service.

Observed runtime: CPython3.12.14, macOS27.0.1 arm64, Torch2.9.1, NumPy2.3.5, Transformers4.57.1, CPU float32, eager attention, eval/inference mode,2 Torch threads. Seed20261115; deterministic algorithms not enabled; no bitwise cross-platform claim. Full causal attention with standard per-call KV caching; no cache shared between calls, eviction, sliding mask or positional-scaling intervention. The configured model window is8192; input cap4096 and output reserve24 are checked before generation, with **no input truncation**. No warmup generations: all90 calls are planned and retained.

Actual complete rendered input ranges: evidence-only107–151;32 records535–557;64 records983–1,005;128 records1,879–1,901 tokens. Template/system/question tokens are included. These are **record-count buckets, not an exact equal-token-length matrix**. Task grammar/evidence counts cause within-bucket differences, and placement comparisons retain their measured lengths rather than assert perfect token equality.

Every evidence record has character offsets and half-open token indices in both plan/results. First evidence-start fractions of complete input are early12.4–20.4%, middle47.4–51.6%, late77.6–88.3%. These differ from nominal10/50/90% record placements because of instruction/template/question overhead and multi-record bundles. All evidence is inside the untruncated full-causal input; presence does not prove the model used it.

## Results with complete denominators

Primary scoring is unchanged `context_probes.score_answer`: exact string equality after stripping surrounding whitespace. No extraction, case folding, semantic judge or human adjudication was substituted. Each entry below is strict correct/planned; every case was eligible and completed, with zero runtime errors or missing responses.

| Records (complete input tokens) | Evidence position | Lookup | Two-hop | Count | Completed / planned |
| --- | --- | --- | --- | --- | --- |
| 32 (535–557) | early | 0/3 | 0/3 | 0/3 | 9/9 |
| 32 (535–557) | middle | 0/3 | 0/3 | 0/3 | 9/9 |
| 32 (535–557) | late | 0/3 | 0/3 | 0/3 | 9/9 |
| 64 (983–1,005) | early | 0/3 | 0/3 | 0/3 | 9/9 |
| 64 (983–1,005) | middle | 0/3 | 0/3 | 0/3 | 9/9 |
| 64 (983–1,005) | late | 0/3 | 0/3 | 0/3 | 9/9 |
| 128 (1,879–1,901) | early | 0/3 | 0/3 | 0/3 | 9/9 |
| 128 (1,879–1,901) | middle | 0/3 | 0/3 | 0/3 | 9/9 |
| 128 (1,879–1,901) | late | 0/3 | 0/3 | 0/3 | 9/9 |

Evidence-only lookup0/3, chain0/3 and count0/3 (9/9 eligible/completed). Overall grid0/81, controls0/9. Output cap hits: **60/90**, including5/9 short controls and55/81 grid cases. All capped responses remain in their planned denominators. The output cap is an answering-contract limitation, not an input overflow or proof of missing context.

Raw short-control examples (post-run diagnosis, **not a replacement score**): lookup-r2 replied `The value of k491648 is 48560430.` for expected `v48560430`, showing a prose/prefix mismatch; lookup-r0 gave12152785 for expected `v32152785`, a wrong value as well as formatting; count-r1 replied `ITEM k529547 has 2 ITEM records.` for expected3; chain controls repeat destination prose and hit the cap. See every [raw response and output token ID](../labs/L3_positions/context_model_artifacts/results.json), including grid failures. There is no claim that semantic rescoring would be zero or that extra prose alone explains all failures.

No uncertainty interval/significance or model ranking: three variants per cell reuse the same small instances across conditions. Uniform strict zeros, failed short controls and frequent capped/formatting failures make this model/prompt/scorer combination uninformative about **why** context use fails or the window at which it degrades. A stronger checkpoint, revised prompt/parser or longer output policy requires a newly frozen protocol and fresh evaluation; this report's zeros stay intact.

## Work, time, memory and replay

90 generation calls consumed **93,696 input tokens and2,011 output tokens**, including EOS where emitted;65.359s summed model-generation intervals,67.033s measured in-run interval including preparation/load/bookkeeping. Preparation and load intervals are separately retained; the external interpreter startup, earlier freeze/tokenizer replay, environment installation and original downloads are outside67.033s. No write/extraction calls, retries or hidden warmup. Tokens are accounting, not equal FLOPs or billed money. Timing is one CPU process/order, not portable throughput or a method comparison. Process-lifetime peak RSS **2,542,534,656 bytes** (macOS units), including imports/load: not per-cell memory, GPU allocation or isolated KV size.

Seven new boundary/control tests and seven existing fixture tests plus nine existing position tests pass (**23 total**), separately from model quality. They cover excluded metadata, cap/reserve rejection, token-span overlap, frozen-plan mutation rejection, strict-score counterexamples, full grid/control coverage and retention of errors/ineligible/capped failures. [Raw tests](../labs/L3_positions/context_model_artifacts/tests.log), [freeze log](../labs/L3_positions/context_model_artifacts/freeze.log) and [run log](../labs/L3_positions/context_model_artifacts/run.log) retain commands' outputs, including Transformers' dtype deprecation warning.

An [independent tokenizer/scorer replay](../labs/L3_positions/context_model_artifacts/verify.py) checks all90 prompt identities, decoded outputs, source/model hashes, evidence spans, unique cases, frozen order, scores and per-cell denominators without loading model weights or regenerating answers. [Verification report](../labs/L3_positions/context_model_artifacts/verification.json) contains exact ranges and hashes. The observed [requirements](../labs/L3_positions/context_model_requirements.txt) describe this reused environment, not a fresh installation or cross-platform hash lock.

From `labs/L3_positions/`, with the pinned local checkpoint and matching environment:

```sh
python -m unittest -v
python context_model_pilot.py freeze --model-dir /path/to/pinned-model --plan /tmp/new-plan.json
# Commit/review the frozen protocol and plan before any model calls.
python context_model_pilot.py run --model-dir /path/to/pinned-model --plan /tmp/new-plan.json --output /tmp/new-results.json
python context_model_artifacts/verify.py --model-dir /path/to/pinned-model --output /tmp/new-verification.json
```

Freeze/run refuse existing output paths; run rejects changed frozen plan/source/model identities. The replay command above verifies the **checked-in artifacts**, not `/tmp/new-results.json`. A rerun is reproducibility evidence, not a fresh independent cohort. No default lab dependency changes.

**Completed:** bounded actual model execution, token/evidence-location accounting, fixed scoring and all-case reporting for the original ARC-3 grid plus short controls. **Still open:** an informative long-context quality curve, prompt/scorer/model qualification, longer/dispersed/realistic tasks and model/position-method comparisons. No PI/YaRN, cache/mask correctness or general effective-window conclusion follows. AGT-2's broader benchmark also remains open; PR#120 is teaching/protocol only. Deck10 slide9 now visualizes this result in EN/ZH, with the earlier grid/questions retained as hidden reading notes and [before/after review captures](../../docs/reviews/architecture-context-model-pilot/README.md).
