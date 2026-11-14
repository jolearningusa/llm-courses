# Four-condition local memory pilot — 2026-11-14 UTC

This is AGT-2's **first executed local answer-model pilot**, separate from PR #120's teaching/protocol and the kit's scripted read-path demo. One pinned SmolLM2-135M-Instruct checkpoint generated **32 answers**: the same eight questions, four types and two authored synthetic profiles in each of no-memory, full-history, verbatim-retrieval and extracted-memory conditions. The full histories fit; all 32 planned pairs were eligible and no inference error occurred.

**The frozen primary result is 0/8 for every condition.** The scorer requires the entire response to equal the requested value after case/whitespace/surrounding-punctuation normalization. Many responses state the correct fact in an extra sentence; other responses are factually wrong. These zeros measure this value-only response contract, not zero factual knowledge, memory equivalence or uselessness of retrieval/extraction. No prompt, scorer or output was changed to improve the result after seeing it.

## Frozen chronology and artifacts

- `37288d2`: [protocol](../labs/kit/memory_pilot_protocol.json), [histories](../labs/kit/tasks/memory_pilot/histories.json), [questions and labels](../labs/kit/tasks/memory_pilot/questions.json) committed before model inference. One predetermined public checkpoint, no search or best seed/checkpoint selection.
- `7f6a6a6`: [standalone runner](../labs/kit/memory_pilot.py) and [eight controls](../labs/kit/tests/test_memory_pilot.py) committed before execution.
- `a7c7e94`: [query-blind writes](memory_pilot_writes_2026_11.json) committed before any scored answer was generated. The separate write process reads only histories, never the question/label file. Each profile's extraction ran once; no semantic correction/deduplication was applied.
- [Answer ledger](memory_pilot_answers_2026_11.json): raw text, complete rendered prompt, selected chunk indices/scores, question/type/profile, reference, exact decision, input hashes, generated token IDs, eligibility, cap hits and measured intervals for every pair. Source/protocol/data/model hashes connect the phases.
- [Model/runtime artifact record](memory_pilot_artifacts_2026_11.json): pinned URLs, hashes, wheel identities, logs and observed [package snapshot](../labs/kit/memory_pilot_requirements.txt). No weights or third-party code are bundled in the repository.

The model/tokenizer revision is **`12fd25f77366fa6b3b4b768ec3050bf629380bac`** of [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac). The 269,060,552-byte `model.safetensors` SHA-256 matches the hub's LFS metadata. All nine selected model/tokenizer/card/config files are hashed; `trust_remote_code=False`, local-files-only loading and offline environment settings were used for inference. No API key, endpoint or paid model service was used.

Runtime: fresh CPython **3.12.14**, macOS **27.0.1 arm64**, Torch **2.9.1**, NumPy **2.3.5**, Transformers **4.57.1**, CPU float32/eager attention, two Torch threads; eval/inference mode. Torch/NumPy seed 20261114, greedy decoding, one beam, one trial and zero retries. Deterministic algorithms were not enabled; there is no cross-platform bitwise-replay claim. The observed package snapshot is not a new course-wide or cross-platform hash lock. Base Torch/NumPy wheels came from the already recorded TRN-4 cohort, then the recorded Transformers wheel set was installed. `pip check` passed.

## Same questions, ceiling and answering contract

All arms use the same system instruction and memory/question wrapper, **512 input tokens maximum** and **16 new answer tokens maximum**. Actual consumed tokens vary and are outcomes, as required by the original worksheet. No input was truncated. Full history is eligible only when the complete rendered prompt fits; the runner returns ineligible before generation when over cap. Extraction has its separately declared **96-new-token write budget**, with the same 512-input ceiling; its cost is additional rather than hidden inside equal answer budgets.

The standalone stores are scoped by profile. Verbatim chunks are seven original turns per profile. Extracted chunks are nonempty unedited model-output lines. Both retrieval paths use **the same lexical overlap score, stopwords, top-k=2 and stable original-order ties**; neither uses embeddings or a reranker. Line lengths/duplicates and source units differ, so this whole-pipeline comparison does not isolate extraction as a single cause or reproduce a semantic vector-store benchmark. All history and reference labels are fictional course-authored fixtures, not LongMemEval data or a human-validated benchmark.

For each question the condition order rotates once; every arm occupies each of the four order positions exactly twice. Each `generate()` starts from its own prompt with a fresh per-call KV cache, while model weights remain loaded; no past-key-value state crosses calls. A one-token arithmetic warmup in each process and setup/load intervals are recorded separately: 32 scored answers + 2 extraction calls + 2 warmup calls = 36 actual model generations. This balancing does not make one local timing sample a portable performance result.

## Primary observations and accounting

“Read in/out” counts actual answer-model input/output tokens; “write in/out” counts extraction-model tokens. Generated EOS tokens count toward output. The two writes are amortized over four questions per profile, **Q=8 total answers per condition**. A future deployment's Q is not assumed.

| Condition | Strict value-only correct / eligible | Read in / out | Write calls; in / out | Total tokens | Write + read/answer seconds |
|---|---:|---:|---:|---:|---:|
| No memory | 0 / 8 | 490 / 106 | 0; 0 / 0 | 596 | 1.243 |
| Full history | 0 / 8 | 1286 / 93 | 0; 0 / 0 | 1379 | 1.308 |
| Verbatim retrieval | 0 / 8 | 685 / 87 | 0; 0 / 0 | 772 | 1.124 |
| Extracted memory | 0 / 8 | 667 / 101 | 2; 301 / 181 | 1250 | 3.417 |

Per eligible returned response (eight per arm, including strict failures), amortized write + read/answer accounting is no memory: 74.500 tokens / 0.1554s; full history: 172.375 tokens / 0.1635s; verbatim: 96.500 tokens / 0.1405s; extracted: 156.250 tokens / 0.4271s. These exclude the separately reported common/phase setup and are not cost per successful answer, portable latency or billed money.

All four question types—updated home, mug preference, updated schedule and object-owner relation—have **n=2 per arm and strict 0/2**. These are eight paired questions within only two profiles, not eight independent sampled users. No confidence interval, significance result or general ranking is claimed. The complete paired ledger is retained rather than pooling unequal subsets.

These token totals are accounting, **not FLOPs, equal compute or API currency**. Non-extracted conditions perform no LLM write/embedding/index-build call; their direct list construction, lexical retrieval and prompt preparation are inside measured read/answer intervals. Extraction includes generation and line-store construction in write time. Both use a trivial in-memory list, not production persistence.

Measured answer-call intervals sum to **4.964s**, and the two extraction/write intervals sum to **2.128s**. Separate process wall times were **6.721s** for write and **6.330s** for answer, including their imports/load, hashing, warmup and bookkeeping. Together these phases took **13.051s**. The difference from per-arm call sums is retained common/phase setup overhead, not attributed to a winning method. Downloads/environment installation are separately observed setup, outside those phase intervals. No billed API cost was incurred; electricity, hardware, model-download and full economic costs were not measured and are not represented as zero.

## Adverse outcomes, without changing the metric

Examples below are **post-run qualitative diagnostics**, not a second accuracy metric or human adjudication:

- Full history answered Mara's home as `Mara's current home is Ghent.` The reference is `Ghent`; the primary scorer rejects the extra prose. This visibly exposes a formatting/scoring confound even when the fact is supplied and named.
- No memory fabricated `New York City` for Mara and `Prague, Czech Republic` for Ivo, despite the abstention instruction.
- Mara's extracted store repeated the mug-color sentence seven times and ended with an unfinished line; it omitted the home, schedule and object-owner information. That store is preserved verbatim, not repaired after inspection.
- Verbatim retrieval answered `Porto` instead of Mara's updated `Ghent`, and answered `Ivo` rather than `Lea` for the violet notebook. Full history is a diagnostic baseline, not a guaranteed ceiling.

One extraction hit its 96-token limit without EOS. Two no-memory and two extracted-memory answers hit the 16-token limit without EOS. All cap hits remain in the planned denominator; hitting the output ceiling is distinct from truncating an over-budget input. Raw logs retain Transformers' `torch_dtype` deprecation warning.

The primary scorer checks a narrow literal contract. **No semantic/LLM judge or human-blinded adjudication was performed.** Raw response/label pairs are reviewable, including false negatives caused by added prose. A later utility/semantic study needs a new predeclared protocol and preferably a fresh cohort; do not silently overwrite this pilot's zeros. This is why the pilot does not close the broader AGT-2 benchmark/human-review/generalization gates.

## Controls and replay

Eight standard-library unit controls exercise strict scoring against negation/extra prose, ranking/ties, complete full history, no-memory exclusion, profile isolation, over-cap rejection without model generation, query/label-blind writing, write mutation rejection and preservation of failed/ineligible planned cases (some tests cover more than one assertion). They pass before the run; [their log](memory_pilot_artifacts_2026_11.json) is retained. Independent tokenizer-only replay verified all **34 write/scored calls** against input counts/hashes and decoded output token IDs, without regenerating answers. The 32 unique question/condition pairs and balanced execution positions were checked independently. There were no model retries or dropped outcomes.

From the repository root, choose fresh environment/output paths:

```sh
python3.12 -m venv /tmp/memory-pilot-env
/tmp/memory-pilot-env/bin/python -m pip install --only-binary=:all: \
  -r llm-agentos-course/labs/kit/memory_pilot_requirements.txt
/tmp/memory-pilot-env/bin/hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-dir /tmp/memory-pilot-model \
  --include README.md config.json generation_config.json tokenizer.json tokenizer_config.json \
  special_tokens_map.json merges.txt vocab.json model.safetensors
/tmp/memory-pilot-env/bin/python -I -m unittest discover \
  -s llm-agentos-course/labs/kit/tests -p test_memory_pilot.py -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/memory-pilot-env/bin/python -I llm-agentos-course/labs/kit/memory_pilot.py write \
  --model-dir /tmp/memory-pilot-model --output /tmp/memory-pilot-writes.json
# Save/commit the write artifact before the next step; do not tune it on questions.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /tmp/memory-pilot-env/bin/python -I llm-agentos-course/labs/kit/memory_pilot.py answer \
  --model-dir /tmp/memory-pilot-model --writes /tmp/memory-pilot-writes.json \
  --output /tmp/memory-pilot-answers.json
```

A replay is a reproducibility observation, not a new independent test cohort. The runner refuses existing output paths and detects changed protocol/history/model/runner/store identities between write and answer phases. Inspect both token/raw-text records and eligibility; do not compare only the top-line score.

## Primary reading and completion boundary

[MemDelta v1](https://arxiv.org/html/2606.29914v1), §3 controlled model/question/history comparisons, §4.4 write-path accounting and partial-coverage limitations, motivates these controls; this is an original tiny course pilot, not reproduction of its tables or its seven strategies. The pinned [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md) documents CPU inference and limitations in accuracy/instruction following. The versioned Transformers generation-doc page was inaccessible to the web reader; actual installed 4.57.1 code and the recorded calls establish the selected generation configuration, without a claim about newer releases.

**Completed:** four implemented standalone input conditions, real fixed-checkpoint answer/extraction calls, matched eligibility, frozen primary scoring and actual write/read cost records. **Still open:** semantic/human-scored utility, larger and adverse cohorts, uncertainty, alternative checkpoints/retrievers, long histories, persistent-store behavior and the broader AGT-2 measured benchmark. PR #120 remains teaching/protocol only; this later pilot is separately dated measured evidence. No student MemoryStore implementation or model-service credential is supplied.
