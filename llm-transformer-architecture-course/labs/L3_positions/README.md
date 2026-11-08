# Lab L3: positions

Used with decks 08, 09 and 10. CPU only, NumPy only.

`positions.py` implements the position methods of Part II side by side: sinusoidal embeddings (deck 01),
RoPE (deck 08), ALiBi slopes and biases (deck 09), and Position Interpolation and YaRN's per-frequency
scaling and temperature (deck 10).

## Run

```bash
python -m unittest test_positions.py
```

## Tasks

1. Prove on paper, then check with `rope()`, that the score between a query at m and a key at n depends only
   on n − m. Which line of `rope()` would break this if you changed it?
2. Print `rope_wavelengths(128)` for base 10,000 and base 500,000 (Llama 3). How many pairs complete less than
   one turn inside an 8K window in each case? Why does a larger base help long contexts?
3. Build `alibi_bias(16, 8)` and plot the softmax of one row for each head. Which heads can still attend
   to the first token from position 15?
4. Use `interpolate()` to extend a 4K model to 32K. What is the smallest gap between two neighboring
   positions now? Explain the short-context regression the PI paper reports.
5. Plot `yarn_ramp(128, 4096)`. Which pairs does YaRN leave untouched, and which does it treat exactly like PI?

## Checks

RoPE scores depend only on the offset (also after interpolation) and rotations preserve norms; the RoPE
frequencies match RoFormer; ALiBi slopes for 8 heads are 1/2 … 1/256 and the bias is linear in distance and
causal; interpolated positions stay inside the trained range; YaRN keeps fast pairs and scales slow pairs
exactly as PI does; and the temperature follows 0.1 ln(s) + 1.

<a id="context-evaluation"></a>
## Controlled context evaluation — audit ARC-3

Paired with deck 10 slide 9 and deck 15 slide 10. Position identities, correct cache/mask execution, and a trained model's ability to use long context need **different oracles**. Passing the existing random-array tests does not establish long-context understanding.

`context_probes.py` adds an original standard-library fixture builder with lookup, two-link tracing and counting tasks. It is inspired by [RULER's task families](https://arxiv.org/html/2404.06654v1); it is **not RULER**, a reproduction of its benchmark, or a model evaluator. Evidence bundles are contiguous; these tiny examples do not reproduce distributed multi-evidence reasoning or realistic long documents.

```sh
python context_probes.py --output my-context-fixtures.json
python -m unittest test_context_probes -v
python -m unittest -v
```

Use a new output name; the command refuses to overwrite an existing file. Default seed 7 and three variants per cell produce **81 fixtures**: three lengths (32/64/128 synthetic records) × three evidence placements (early/middle/late) × three task types × three variants. The summary reports 81 debug-oracle agreements and **zero model runs**. That is fixture validation, not 100% model accuracy. The new builder is standard-library only; the existing position tests still need NumPy.

### What the fixture checks actually establish

For a fixed task/variant, moving evidence preserves its question, expected answer and complete filler sequence. Length adds filler records. IDs and expected-answer metadata stay outside `prompt_for(case)`; only its context/question text may be sent to a future model adapter. The answer is intentionally present or derivable from the context, but the manifest's grading labels must not be appended to prompts. `solve_fixture` is a grammar-specific debugging oracle, not an LLM. Tests include a literal chain and a changed-evidence control that the expected-answer check rejects.

**Record counts are not model token counts.** The fixture manifest is a construction exercise. Before making a model length/position claim, use a pinned tokenizer and chat template to record the complete input length, evidence token indices and output reserve. Rebuild/pad a model-token matrix if needed, and report any deviations. Do not label the 128-record fixture “128K context.” The slide bars have normalized positions and are not byte or memory-size diagrams.

### Plan the actual model experiment

1. Freeze checkpoint/tokenizer/template revisions, positional-scaling configuration, attention backend/mask, precision, generation limits, decoding settings and scoring rules. Keep the question, answer and filler identities fixed for paired placement comparisons. Distinguish deliberate mask/window changes from a pure position comparison.
2. Choose token lengths that fit the supported execution contract with template and generation headroom. Use early/middle/late evidence and more than one task type. Declare which evidence is directly visible under each mask; a cache-equivalence pass does not imply old facts remain useful after eviction. A configured maximum window is not a quality threshold.
3. Run every planned eligible case with a fixed trial/seed policy. Retain overflows, refusals, truncations, parse failures and missing runs. Score with a frozen task-specific oracle. The teaching scorer uses exact string match after stripping surrounding whitespace; define any production answer parsing before observing results.
4. Report each length × placement × task cell with eligible/attempted/completed counts, quality, uncertainty and failures. Also record total tokens, latency and memory with a declared boundary. Avoid hiding a failed middle-position cell behind one aggregate score. Three classroom variants are not a claim of statistical power.
5. Keep numerical identities, implementation tests and learned-quality outcomes in separate rows. Without a model/adapter run, mark quality and performance **not measured**. No API credentials, model calls, downloads or target benchmark are performed by this fixture builder.

| Evidence | Oracle | Current scope |
|---|---|---|
| Position mathematics | Relative-position identities / norm preservation | Existing L3 NumPy checks |
| Execution with a declared mask | Cached versus recomputed outputs within tolerance | Existing L5 checks; random weights |
| Fixture construction | Preserved facts, coverage, debug oracle and rejected wrong answer | New seven-test fixture suite |
| Learned context quality | Frozen trained checkpoint, task outputs and per-cell scoring | Not measured by this release |

### Predict and submit

A model fits a long prompt and passes one passkey lookup. Does that establish multi-hop tracking or aggregation across that window? **No.** Likewise, a streaming implementation may reproduce its intended windowed computation while failing a task that needs older evidence. Diagnose the workload and available information before calling that a cache bug.

Submit the manifest/seed, prompt-construction boundary, your planned tokenized matrix, preserved paired controls, one rejected answer, and a report with explicit not-run cells. If you execute a model, add raw outputs and all failure denominators. The course does not supply a model adapter or measured model scores here.

Retain deck 10's mathematical checkpoints: PI maps token 20,000 to 2,500 when extending 4,096 to 32,768; derive YaRN's temperature for scales 16/32 and explain pre-rotation key caching for dynamic scaling. Retain deck 15's questions on theoretical receptive span, rolling-buffer slots, sink eviction and caching before RoPE. The new plan complements those implementation questions rather than treating them as comprehension tests.

中文提示：长度 × 证据位置 × 任务类型要分别记录结果。这里生成的是“记录数”网格，不是 token 长度网格；81 个调试判定通过不等于模型准确率 100%。真实模型测试需固定分词/模板、保留失败和缺失，并区分位置数学、缓存正确性与任务质量。
