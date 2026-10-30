# Appendix A — LLM systems projects and execution plans

Choose one measurable systems question, establish a credible baseline, then implement the smallest change that can answer it. This appendix offers **26 independent projects**, including engineering capstones, reproducibility studies and research hypotheses. P01–P20 retain the IDs in the [original catalogue](POTENTIAL_PROJECTS.md); P21–P26 are extensions. None is a claim of an implemented product or a guaranteed speedup.

The supplied *LLM Inference System: Project Ideas and Technical Strategy* motivates a coherent longer-term product: an optimizer that profiles a workload, chooses an optimization layer, evaluates candidates and retains validated results. A class project should implement one evidence-producing component or one narrow decision experiment. Python is a convenient first controller; Rust is an optional implementation choice after the interfaces work. Existing compilers, libraries, runtimes and serving engines remain dependencies.

Read the [paper and repository review](../references/project_research_review.md) for prior art and the [deck review](../notes/project_appendix_review.md) for teaching links. Literature reviewed 2026-10-29; P26 and its bottleneck review added 2026-10-30. Research labels below mean testable hypotheses; novelty requires a new related-work check at project start.

## A1. Select a project

**E** = engineering; **R** = research hypothesis; **Rep** = reproduction/measurement. Durations assume two students with the listed prerequisites, approximately 8–12 hours each per week. They are planning estimates, not tested student pacing. GPU-hour caps below are initial allocations; W1 pilots must verify fit and cost. Do not purchase hardware on these estimates alone.

| ID | Project and type | Minimum target / duration | Course connection |
|---|---|---|---|
| [P01](#p01) | Cross-engine inference benchmark · E/Rep | One supported device, two engines; 4 weeks | Decks 03, 15–17, 35; L7–L8 |
| [P02](#p02) | Serving configuration autopilot · E/R | One serving GPU; 6 weeks | Decks 17, 32; L8 |
| [P03](#p03) | Agent lifecycle and KV policy · R/Rep | CPU model, then one serving GPU; 6 weeks | Decks 16, 21; L7, L10 |
| [P04](#p04) | KV handoff and split serving · E/R | Two compatible GPUs and connector; 6 weeks | Deck 18; L8 |
| [P05](#p05) | Specialized-engine integration feasibility · E | Source access; supported DeepInfer target for runtime; 4 weeks | Decks 15, 32, 35 |
| [P06](#p06) | Compiler/library path selection · E/R | CPU first; GPU for GPU claims; 6 weeks | Decks 05–06, 22, 34; L2, L11 |
| [P07](#p07) | Kernel abstraction comparison · Rep | One GPU supporting two chosen paths; 6 weeks | Decks 07–12, 33; L3–L5 |
| [P08](#p08) | Verified kernel-search agent · E/R | One GPU plus optional generation API; 6 weeks | Decks 08, 32; L3–L5 |
| [P09](#p09) | Automated inference optimization engine · R | One device/model/engine; 8 weeks | Decks 30, 32; capstone |
| [P10](#p10) | Precision and model-quality gate · E/Rep | CPU quality reference; eligible GPU recipe; 6 weeks | Decks 13–14; L6 |
| [P11](#p11) | MoE routing/skew workbench · E/R | One GPU; 2–4 for exchange; 6 weeks | Deck 19; L4, L9 |
| [P12](#p12) | Training layout and overlap explorer · E/R | CPU model; 2–8 GPUs for measured layouts; 6 weeks | Decks 25–28; L12–L13 |
| [P13](#p13) | Recoverable-training benchmark · E | CPU first; optional cluster; 4 weeks | Deck 29; L14 |
| [P14](#p14) | Export and edge-backend optimizer · E/R | One supported CPU/edge target; 6 weeks | Decks 22–23, 34; L11 |
| [P15](#p15) | Accelerator software maturity/portability · E/Rep | Source dossier; NVIDIA + AMD for measurements; 6 weeks | Decks 24, 33–34; L15 |
| [P16](#p16) | Host-to-device pipeline optimizer · E | CPU + one GPU; 4 weeks | Decks 03–04; L1 |
| [P17](#p17) | Adaptive speculative decoding · R/Rep | CPU probability tests; GPU draft/target; 6 weeks | Deck 20; L10 |
| [P18](#p18) | Domain-transfer deployment study · E | One non-LLM model/device; 4 weeks | Decks 23, 34; L11 |
| [P19](#p19) | Reproducible optimization workbench · E | CPU orchestration; two real experiment adapters; 4 weeks | Decks 03, 31–32 |
| [P20](#p20) | Rollout/trainer resource balance · R | CPU model; 2–8 GPUs for a real loop; 6 weeks | Decks 20, 25, 30; L15 |
| [P21](#p21) | Search-history transfer and stopping · R | Saved trials, then one target; 8 weeks | Decks 06, 30, 32 |
| [P22](#p22) | Adversarial numerical validator · R | CPU reference + one GPU; 6 weeks | Decks 08, 13, 32 |
| [P23](#p23) | Cache-operation identity and restoration · R | CPU state model; eligible AKV/engine target; 8 weeks | Decks 16, 21; L7, L10 |
| [P24](#p24) | Distributed candidate evaluation · R/E | Two GPUs, homogeneous communicator; 8 weeks | Decks 19, 26; L9, L12 |
| [P25](#p25) | Heterogeneous edge multi-model scheduling · R/E | One supported edge board; 8 weeks | Decks 04, 23, 34; L11 |
| [P26](#p26) | GPU selection and decoding control · E/R | One supported GPU; 6 weeks | Deck 36; L1, L8, L10 |

**Recommended starting choices:** P01/P06/P13 establish evidence with modest setup. P03 is the closest AgentOS continuation. P14/P25 fit edge-device experience. P09 is the proposed product's research core, after a working evaluator. P21/P22 are narrower research studies with cleaner ablations. P04/P11/P24 require allocations confirmed before enrollment.

## A2. Turn the strategy into independent projects

| Supplied strategy | Implementable course project | What constitutes progress |
|---|---|---|
| A. Automated inference optimization | P09, optionally P21 | Better feasible outcomes or lower search cost against budget-matched baselines |
| B. Kernel generation/tuning | P07, P08, P22 | Correct implementations, stronger validation and measured value over current libraries/compilers |
| C. Accelerator-vendor optimization | P14, P15, P25 | One supported target, explicit missing capabilities and a validated deployment improvement |
| D. Performance harness | P01, P19 | Independently reproducible experiments and failure accounting used by real studies |
| E. New serving engine | P03–P05, P17 | A narrow policy/adapter gap tested against an existing engine; no full-engine rewrite in the MVP |

A dependency is an interface, not an obligation to implement another project. P09 may consume a minimal independent evaluator instead of waiting for P19. P21 can begin with archived trials but needs new target trials before a hardware-performance claim. P23 is a state-protocol study; it does not require a new cache policy from P03.

## A3. Common experimental and delivery contract

Freeze the model/tokenizer revision, target architecture and memory, driver/runtime/compiler/library versions, input shapes/layouts/dtypes, sampling settings, workload distribution and quality bounds. Record the actual dispatch path and unsupported cases. Keep the course's historical CPU lab environments separate from newly qualified environments.

**Workload pilot.** Begin with a decoder small enough to fit weights, workspace and KV on the selected device. For serving, start with prompt lengths 128/1,024/4,096 and generation caps 32/128, concurrency 1/4/16, and prefix overlap 0/50/90%; these are proposed fixtures, not a benchmark standard. Reduce the grid after the fit pilot. Pin exact token IDs and actual generated lengths. Trace replay preserves offered arrivals and dependencies; replay with fixed output lengths tests systems behavior, while live generation separately tests task quality. Never infer live task success from a fixed trace.

**Split before search.** Separate tuning, validation and final test by task/prompt family or shape family, not just random repetitions of the same input. For small studies use a declared 60/20/20 split with an entire long-context or burst regime held out. No-history, random/grid, current defaults and an eligible existing tuner are the minimum search controls. If a dependency cannot run, record it and narrow the comparison claim.

**Timing and accounting.** Separate search, compile, load, warm execution and end-to-end time. Include failed builds, wrong candidates, timeouts, process restarts, generation tokens/API cost, human interventions and occupied device time. Preserve raw samples. Run at least five independent search seeds for search comparisons; collect three or more independent measurement runs per final configuration and report uncertainty. For p99 claims, choose enough completed requests to estimate the tail and publish the population; a tiny sample should report individual observations instead. Randomize candidate order to reduce thermal/order bias. Profiler runs diagnose; unprofiled runs establish performance.

**Correctness and quality.** Check tensor shape/dtype/layout, complete outputs, finite values and operation-specific numerical tolerances; required gradients for training. Compare against a stronger-precision reference where feasible. Add outliers, zeros, near-zero values, tails, masks and mutation/alias tests appropriate to the operator. For changed precision, attention context or sampling semantics, run held-out task-quality tests with predeclared noninferiority bounds. Testing is evidence, not a proof for arbitrary inputs.

**Serving metrics.** TTFT runs from request submission through first visible token; report transport/queue boundaries. TPOT averages intervals after the first output token, excluding single-output requests and reporting their count; retain inter-token distributions separately. Define goodput as completed requests satisfying the declared quality and latency constraints per observation second. Report offered, accepted, rejected, completed, failed and unfinished requests; drain queues or disclose censoring. Agent studies additionally report task completion latency/success and per-tenant fairness.

**Decision rule.** In W1 measure noise and choose a practical improvement threshold before final evaluation. A suggested engineering gate is at least 5% end-to-end improvement with an uncertainty interval excluding zero, no violated quality/SLO/memory bound and a plausible amortization horizon. This is a proposed project threshold, not a promise or a grading requirement. Search research instead compares time to a fixed feasible target and best feasible outcome versus total budget. An experiment that rejects its hypothesis is a valid capstone.

**Reproducibility package.** Deliver a one-page proposal, pinned environment, rerun commands, workload manifest, immutable result archive, correctness/quality fixtures, raw traces/samples, one adverse case and one ablation, a report and a short presentation. Include a last-known-good configuration and rollback instructions. Generated candidates run in bounded workers and cannot modify the evaluator or acceptance policy. These are project requirements, not infrastructure already shipped by the appendix.

## A4. Detailed project plans

<a id="p01"></a>
### P01 — Cross-engine inference benchmark

**Question and scope.** Can two engines be compared without hiding differences in tokenization, quantization, batching or caching? Build an evaluator and a mechanism report, not an engine ranking website. Use two engines that support the same model/computation on one target. If only a GGUF quant and a different checkpoint are feasible, label the experiment a deployment tradeoff and add quality evaluation; it is not a controlled engine-speed comparison.

**Execution — 4 weeks.**

1. **Week 1.** pin one small model, two engines and token IDs; reproduce each engine's own benchmark and expose first/last-token events.
2. **Week 2.** build open-loop arrival replay and accounting tests, including a timeout, cancellation and a one-token response.
3. **Week 3.** run cold, shared-prefix and early-mismatch cases at low/medium/high offered load; match total device resources and explicit cache state.
4. **Week 4.** rerun held-out prompts and bursts, explain one difference with a trace, package adapters and raw events.

**Baselines and evidence.** Compare adapter metrics against engine-native counters and an independently recomputed event log. Report TTFT/TPOT, goodput, memory, actual generated lengths, unfinished work and tokenizer/quality differences. [SGLang](../references/project_research_review.md) supplies an existing prefix-aware baseline; deck 35 supplies the engine vocabulary, not a performance prediction.

**Budget / stop-go.** Proposed cap: 12 device-hours after setup, no paid generation service. Deliver two adapters, workload fixtures and a comparability table. Stop a claimed ranking if identities cannot match; the useful deliverable then becomes the explanation of why they differ.

<a id="p02"></a>
### P02 — Serving configuration autopilot

**Hypothesis.** A cost-aware search finds an SLO-feasible configuration with less tuning time than unprioritized search. Restrict the MVP to one engine's supported sequence limit, batched-token budget and one cache/admission setting. Keep model precision fixed.

**Execution — 6 weeks.**

1. **Week 1.** establish defaults, three arrival regimes and fit/OOM boundaries.
2. **Week 2.** integrate the pinned engine's existing tuner, random/grid search and one common evaluator.
3. **Week 3.** add a feasibility-first planner that uses measured memory and iteration time to rank at most 24 configurations.
4. **Week 4.** search tuning traces using equal occupied-GPU-time caps and at least five seeds; archive rejected configurations.
5. **Week 5.** freeze candidates and test unseen burst/prefix mixtures.
6. **Week 6.** repeat the frontier and export a configuration with rollback instructions.

**Baselines and evidence.** Current defaults, [vLLM auto-tune](https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md), random search and a simple hand-tuned setting. Existing [Sarathi-Serve](../references/project_research_review.md) mechanisms prevent claiming chunked prefill itself as novelty. Plot feasible goodput against search time, tail latency, memory and failure rate. Ablate memory filtering and ranking separately.

**Budget / stop-go.** Suggested total 30 device-hours; reduce the candidate set after W1 if startup dominates. Proceed as research only if budget-matched held-out results beat a credible tuner or explain a new failure regime. Otherwise deliver a reproducible tuning integration. No live production reconfiguration in the MVP.

<a id="p03"></a>
### P03 — Agent lifecycle and KV retention policy

**Hypothesis.** Retention decisions conditioned on tool-duration uncertainty and shared memory pressure improve completed agent tasks over fixed TTL. This project preserves the attended computation: retain/offload/recompute exact state. Dropping context is a different quality-changing intervention studied in P23.

**Execution — 6 weeks.**

1. **Week 1.** reproduce the L10 single-wait reference and define multi-turn task traces with waits, branches, cancellation and model-version IDs.
2. **Week 2.** add shared capacity/queue simulation and calibrate prefill/transfer costs on the selected engine if available.
3. **Week 3.** implement LRU, fixed TTL and a Continuum-inspired duration-aware baseline.
4. **Week 4.** implement one confidence/pressure-aware decision rule through supported engine hooks; keep the state model separate.
5. **Week 5.** replay short/long/heavy-tailed tool waits under several loads; inject wrong estimates and early returns during offload.
6. **Week 6.** evaluate held-out workflows, live quality where applicable and fairness.

**Baselines and evidence.** [Autellix and Continuum](../references/project_research_review.md) already study program scheduling and TTL. Measure task p95 completion, success, resume TTFT, recomputed tokens, transfer bytes, idle byte-seconds and displaced work. Ablate pressure input and duration uncertainty. Test stale weights/adapters and shared-owner cancellation.

**Budget / stop-go.** Start with 20 device-hours; if no supported engine hook exists, finish a calibrated simulator plus feasibility report and make no engine-performance claim. Deliver trace schema, policy module, identity tests and explicit null/adverse regimes. Novelty must be narrower than “cache-aware agents.”

<a id="p04"></a>
### P04 — KV handoff and split-serving testbed

**Question.** Under what measured transfer/queue conditions does separating prefill and decode help? Choose one supported model, engine connector, KV layout and pair of devices. Universal cross-engine KV serialization is outside the MVP.

**Execution — 6 weeks.**

1. **Week 1.** qualify the connector and derive actual cache bytes, including metadata and parallel layout.
2. **Week 2.** instrument producer-ready, transfer-start/end, consumer-ready and first output events.
3. **Week 3.** compare two colocated workers with one prefill/one decode worker under the same two-device reservation; sweep short/long prompts and load.
4. **Week 4.** add cancellation, transfer failure and consumer restart; check ownership and reclaim.
5. **Week 5.** test link contention and an unfavorable small-request regime.
6. **Week 6.** hold out a workload mix and reconcile measured request time with the stage ledger.

**Baselines and evidence.** [DistServe and Mooncake](../references/project_research_review.md) establish phase separation and KV-oriented placement. Use the current engine's supported path as the baseline and compare transfer against recomputing the prefix. Validate numerical/sampling contracts, failed/unfinished accounting and cache ownership. Report goodput, both latency SLOs, transfer bandwidth achieved and CPU/network occupancy.

**Budget / stop-go.** Proposed cap 24 paired-device hours, meaning 48 device-hours. Confirm two devices and connector compatibility before W2; otherwise deliver source/analytical work. Ship deployment manifests, event fixtures and a measured break-even map. Reject the split if handoff/queues erase the gain.

<a id="p05"></a>
### P05 — Specialized-engine service integration feasibility

**Question.** Is there a concrete service gap worth filling around a specialized engine such as DeepInfer? The outcome may be a decision to use an existing serving system instead. Do not infer missing internal KV management from a minimal offline demo.

**Execution — 4 weeks.**

1. **Week 1.** pin [DeepInfer](https://github.com/deepseek-ai/DeepInfer), inspect supported models/targets, entry points, execution/state ownership and licensing. Record actual source symbols.
2. **Week 2.** compare streaming, batching, cancellation, queue bounds, observability and recovery with an existing service framework; define one missing interface.
3. **Week 3.** on a supported target, implement only that adapter with bounded queues and cancellation. Without target access, write executable interface mocks explicitly labeled as mocks.
4. **Week 4.** inject disconnect/load/restart cases and evaluate whether the adapter adds useful behavior without duplicating upstream.

**Baselines and evidence.** Original engine/offline path plus a supported vLLM/SGLang service with comparable model support if possible. Compare integration effort, capability coverage, output/event correctness and resource cleanup. Performance requires real target execution; mocks establish interface behavior only.

**Budget / stop-go.** Two weeks of source/compatibility work is the hard gate before allocating a large model. Runtime budget follows the supported deployment, not an invented laptop substitute. Deliver a go/no-go dossier, capability matrix and either a tested narrow adapter or a target-access plan. Stop if required APIs/hardware are unavailable or upstream already supplies the feature.

<a id="p06"></a>
### P06 — Compiler and library path selector

**Hypothesis.** A small shape-aware selector improves deployment cost by choosing among existing optimized paths before requesting custom code. Use one decoder block and one trace-dominant operator. The initial candidates are eager, Inductor and an eligible AOT/library path; they are alternatives, not a mandatory serial chain.

**Execution — 6 weeks.**

1. **Week 1.** capture model/operator contracts and a baseline profile for prefill/decode shapes.
2. **Week 2.** inspect graph breaks, guards, library dispatch and layout conversions; build compatible artifacts.
3. **Week 3.** measure cold compilation/load and warm timing across a finite shape grid.
4. **Week 4.** implement a rule/table selector including “use default” and unsupported-input fallback.
5. **Week 5.** test held-out ragged/dynamic shapes and a fresh-process loader.
6. **Week 6.** compare full-block/request gains with operator predictions and quantify amortization.

**Baselines and evidence.** Eager plus current compiler defaults and eligible tuned vendor libraries, not eager alone. [Ansor, Mirage and the runtime-share preprint](../references/project_research_review.md) provide comparison questions about search and system impact. Preserve outputs and required gradients; reject wrong dtype/stride/shape contracts. Measure dispatch overhead, compile cache misses and artifact size as well as latency.

**Budget / stop-go.** CPU MVP; proposed GPU extension 20 device-hours. Deliver a per-shape decision table, artifacts/loaders, trace evidence and fallback tests. If library calls dominate and no end-to-end gain survives overhead, report the bound and retain the default. Do not promise universal automatic graph conversion.

<a id="p07"></a>
### P07 — Kernel abstraction comparison

**Question.** How do two supported programming paths trade implementation effort, compilation cost and performance across shapes? Choose Triton/TileLang, CUDA/CuTe or another eligible pair on one target. Two kernels are sufficient: a reduction and ragged/grouped GEMM. Source portability is not performance portability.

**Execution — 6 weeks.**

1. **Week 1.** qualify both toolchains and freeze library references, input/gradient contracts and tuning budgets.
2. **Week 2.** implement the reduction including tails and numerical edge cases.
3. **Week 3.** implement the matrix/grouped case with empty groups and skew.
4. **Week 4.** tune each with equal measurement budgets; inspect generated code, register/shared-memory use and one confirming profile.
5. **Week 5.** test unseen shapes and at least one case where the library wins.
6. **Week 6.** integrate one winner into its surrounding computation and report total cost.

**Baselines and evidence.** Current vendor library/compiler path and untuned versions of both implementations. [KernelBench and TileLang](../references/project_research_review.md) motivate the comparison without making a language winner inevitable. Keep exact math/precision policy aligned and test required gradients. Include sanitizer diagnostics where available, compilation failures, human development time and per-shape results.

**Budget / stop-go.** Proposed 24 device-hours, at most 32 configurations per kernel/path after pruning. Deliver editable source, build locks, tuning logs and an implementation-effort diary. Stop broadening the language set when two paths reproduce; one rigorously explained loss is more useful than several incomplete ports.

<a id="p08"></a>
### P08 — Verified kernel-search agent

**Hypothesis.** Execution/profiling feedback improves a bounded operator-search process under equal total cost. One operator family, a limited edit region and an independent evaluator are the MVP. An LLM is one candidate proposer; hand templates and existing autotuners remain competitors.

**Execution — 6 weeks.**

1. **Week 1.** select 8 tuning and 4 held-out shape cases and reproduce the library/compiler baseline.
2. **Week 2.** build bounded compile/run workers, immutable trial records and deliberately incorrect fixtures.
3. **Week 3.** connect random/template search and an existing compatible tuner; freeze feedback fields.
4. **Week 4.** add a generation/refinement loop restricted to the operator template.
5. **Week 5.** compare feedback/no-feedback and template/LLM search with five seeds under equal occupied-device and total wall-time limits.
6. **Week 6.** rerun winners in a clean process and one full-block workload.

**Baselines and evidence.** Reuse [KernelBench or FlashInfer-Bench](../references/project_research_review.md) contracts where applicable; do not present their evaluation loop as newly invented. Track correctness rate, best feasible speed, time to target, generation tokens/cost, compile failures and manual intervention. The agent cannot edit tests, timing or acceptance thresholds. Corrupt outputs, wrong masks and partial writes must fail.

**Budget / stop-go.** Proposed 30 device-hours; set an explicit API-money cap in the proposal and offer template-only operation. Deliver source, evaluator adapters, all trials and budget curves. Go forward only when independent re-evaluation confirms the winner; otherwise analyze failure trajectories.

<a id="p09"></a>
### P09 — Automated inference optimization engine

**Hypothesis.** Choosing among optimization layers gives better SLO-feasible outcomes per total search budget than always tuning serving settings or kernels. This is the supplied strategy's main product direction, reduced to a research prototype. Existing [KernelEvolve, Ansor and FlashInfer-Bench](../references/project_research_review.md) substantially overlap the architecture.

**Execution — 8 weeks.**

1. **Week 1.** freeze one model/device/engine, three workload regimes, quality/SLO bounds and an end-to-end bottleneck profile.
2. **Week 2.** make one evaluator independently reproduce baseline results.
3. **Week 3.** provide typed adapters for supported serving parameters and compiler/library choices.
4. **Week 4.** add one narrowly supported operator family only if profiling justifies it.
5. **Week 5.** implement a planner ranking expected system benefit against compile/search cost and feasibility.
6. **Week 6.** compare defaults, random allocation, best single-family tuner and a cost-aware heuristic under equal budgets.
7. **Week 7.** freeze and test a held-out regime plus composition of accepted changes.
8. **Week 8.** reproduce selected artifacts in a fresh environment and present a deployment/rollback decision.

**Evidence.** Performance-versus-total-search-budget curves, SLO frontier, memory/quality, failure rate, planner overhead and human effort. Ablate layer choice, cost estimates and history independently. Operator speedup must reconcile with observed whole-request improvement. A composition is a new candidate, never automatically accepted.

**Budget / stop-go.** Proposed 40 device-hours after a fit pilot; no more than three action families. A suggested research target is 20% less time to a predeclared feasible target at matched quality, subject to uncertainty; freeze the actual target after W1. If layer choice does not help, deliver a negative result and the reusable evaluator. No new driver/compiler/runtime or autonomous production rollout.

<a id="p10"></a>
### P10 — Precision and quality gate

**Question.** Does a supported precision recipe improve the full workload within a declared quality bound after packing, scales, conversions and fallback? Choose exactly one of weight/activation/KV precision for the MVP.

**Execution — 6 weeks.**

1. **Week 1.** pin one model/recipe/target and split calibration from evaluation prompts.
2. **Week 2.** compute and inspect actual packed bytes, metadata, padding, workspace and residual full-precision state.
3. **Week 3.** implement output/error tests plus long-context or outlier cases; calibrate quality bounds before optimization.
4. **Week 4.** verify actual accelerated dispatch and measure conversion overhead separately.
5. **Week 5.** evaluate end-to-end latency, memory and held-out task quality across batch/context regimes.
6. **Week 6.** rerun an adverse regime and publish a support/quality/performance dossier.

**Baselines and evidence.** Reference precision, an existing supported quantizer/kernel and the same recipe without optional tuning. [KIVI](../references/project_research_review.md) supplies an established asymmetric KV comparison when that recipe is selected; it is not a requirement to implement two-bit kernels on unsupported hardware. Include tolerances, quality uncertainty, fallback frequency and total bytes. Training extensions additionally check updates/progression; AMP, PTQ and QAT are different experiments.

**Budget / stop-go.** Suggested 24 device-hours plus a fixed quality-evaluation token budget. Deliver calibration/evaluation identities, recipe manifest, quality fixtures and actual dispatch evidence. A byte reduction with no supported fast kernel remains a capacity result; reject a speed claim if conversion or quality loss invalidates it.

<a id="p11"></a>
### P11 — MoE routing and skew workbench

**Hypothesis.** Shape/skew-aware execution improves a fixed routing contract over one grouped-compute configuration. Begin with local routing/permute/GEMM/combine, then add one supported dispatcher. Do not change expert choices or drop tokens to manufacture a gain.

**Execution — 6 weeks.**

1. **Week 1.** build deterministic top-k/gate fixtures, including uniform, hot-expert and empty-expert cases with equal total assignments.
2. **Week 2.** validate permutation, weighted combination and required gradients against a reference.
3. **Week 3.** compare eligible grouped library paths on one device.
4. **Week 4.** if available, instrument dispatch/expert/combine on 2–4 devices with fixed placement.
5. **Week 5.** test one policy such as a shape-based local kernel choice; vary skew and arrival batches.
6. **Week 6.** hold out route distributions and reconcile maximum-rank phase time with full iteration/request time.

**Baselines and evidence.** Pinned engine/library implementation and fixed grouped kernel. DeepGEMM/DeepEP/AMD AITER are candidates only when the chosen model, datatype and target are supported; see the [course bibliography](../references/bibliography_and_links.md). Measure assignment counts, routed bytes, per-expert sizes, load imbalance, local compute and exposed communication. Ablate policy and preprocessing overhead.

**Budget / stop-go.** Proposed 20 single-device hours; optional 12 hours on a 2–4 device allocation. Deliver route fixtures, a phase trace and policy/dispatch manifest. Without distributed access claim only local results. Stop custom compute tuning when the maximum-rank exchange dominates and document the changed bottleneck.

<a id="p12"></a>
### P12 — Training layout and overlap explorer

**Question.** Can a small planner reject infeasible layouts and identify one that improves useful training throughput? Fix model, optimizer, effective batch, data progression and numerical recipe. Use two supported layouts; do not build a universal parallelism compiler.

**Execution — 6 weeks.**

1. **Week 1.** reproduce a small TorchTitan/Megatron or course training baseline.
2. **Week 2.** build persistent/transient/activation/workspace memory accounting and topology constraints.
3. **Week 3.** validate two feasible layouts' updates and observed peak memory.
4. **Week 4.** measure critical-path collectives and bucket/readiness overlap; change one layout or recomputation choice.
5. **Week 5.** include an apparent-fit case that exceeds transient memory and a communication-unfriendly placement.
6. **Week 6.** test a held-out sequence/microbatch setting and compare planner predictions with measurements.

**Baselines and evidence.** An eligible maintained training configuration, a simple memory-only planner and the measured alternate layout. Keep global batch/token counts constant when changing accumulation. Record maximum-rank step time, peak allocated/reserved memory, exposed wait, compile/setup and useful tokens over wall time. The source walkthrough in L13 supplies traceable configuration symbols, not a cluster-speed estimate.

**Budget / stop-go.** Confirm 2–8 GPUs before W3; proposed 12 cluster-hours at the selected size and a checkpoint/recovery allowance. Deliver rank maps, feasibility exclusions, update tests and prediction residuals. A CPU-only study remains analytical/source work with a falsifiable hardware protocol. Stop expansion beyond two layouts when numerical equivalence or runtime support is unresolved.

<a id="p13"></a>
### P13 — Recoverable-training benchmark

**Question.** Does a checkpoint optimization increase useful progress after interruption without silently changing the training trajectory? CPU execution is sufficient for a rigorous local MVP; distributed durability is a separate extension.

**Execution — 4 weeks.**

1. **Week 1.** reproduce uninterrupted versus resumed training with model, optimizer, scheduler, RNG and sampler/data position.
2. **Week 2.** inject missing RNG/data state, an incomplete file and a crash before snapshot publication; ensure each fails or diverges visibly.
3. **Week 3.** compare synchronous checkpoints with one bounded improvement such as an immutable async snapshot on supported storage.
4. **Week 4.** vary interruption position and checkpoint interval, rerun determinism checks and report useful tokens/updates over total time.

**Baselines and evidence.** The L14 exact-resume reference, no-interruption execution and the same process with synchronous checkpointing. Measure snapshot capture, persistence/restore, lost/repeated work and memory overhead. An async writer must not observe parameters mutated by the next step. Local atomic rename is not proof of distributed commit or power-loss durability.

**Budget / stop-go.** One workstation, no accelerator required; proposed optional cluster extension 8 allocation-hours. Deliver failure fixtures, state manifest, publication/recovery protocol and a progress ledger. Retain synchronous operation if buffering or consistency costs erase the gain. An identified silent-resume bug is a valid engineering result even without speed improvement.

<a id="p14"></a>
### P14 — Export and edge-backend optimizer

**Hypothesis.** Choosing a supported partition/shape/packing configuration improves deployment latency or footprint over the target's default optimized path. Start with one model and device; AOTInductor and ExecuTorch are different artifact/runtime paths, not interchangeable native formats.

**Execution — 6 weeks.**

1. **Week 1.** qualify model operators, SDK, device, export guards and numerical reference.
2. **Week 2.** build producer and fresh-process loader; archive artifact/environment identity.
3. **Week 3.** map delegated partitions, fallback operators, transfer/synchronization and memory plan.
4. **Week 4.** change one allowed partition boundary, shape specialization or packing choice; validate outputs and unsupported-input rejection.
5. **Week 5.** measure cold compile/load, first request, warm latency, peak memory and sustained operation on held-out shapes.
6. **Week 6.** compare against the vendor/default optimized path, calculate amortization and package deployment/rollback.

**Baselines and evidence.** [ExecuTorch's eligible backend](https://docs.pytorch.org/executorch/stable/backends-overview.html), CPU/default delegate and vendor SDK baseline where accessible. Include CPU fallback and boundary costs; a high delegated-node percentage is not itself a speedup. Use the L11 CPU artifact as a reference exercise, not evidence for a phone NPU.

**Budget / stop-go.** One supported device; proposed 24 target-hours with no thermal/power claim unless telemetry is available. Deliver artifact compatibility/partition matrix, loaders and raw cold/warm records. If SDK access or custom operators block export, produce a precise feasibility dossier and stop before adding another target.

<a id="p15"></a>
### P15 — Accelerator software maturity and portability dossier

**Question.** Which missing software capability prevents a real workload from using an available accelerator efficiently? This makes the supplied chip-vendor idea concrete. Compare responsibilities across framework integration, compiler/lowering, libraries, runtime, profiling and collectives; do not equate a graphics driver with an AI stack.

**Execution — 6 weeks.**

1. **Week 1.** select one inference dataflow and model contract; identify required operations/formats/shapes.
2. **Week 2.** map NVIDIA and AMD equivalents plus missing target features using primary support matrices and source symbols.
3. **Week 3.** run allocation/copy/one anchor operation and a small whole-model path on each available target.
4. **Week 4.** identify actual CPU fallback, layout conversion or untuned library selection.
5. **Week 5.** fix one supported gap or tune one eligible path, including existing TunableOp/hipBLASLt choices.
6. **Week 6.** revalidate numerical/quality/performance on held-out shapes and record porting effort.

**Baselines and evidence.** Current eligible vendor library/compiler paths, not unoptimized handwritten code. [AMD optimization guidance](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) establishes existing tuning. Report supported/unsupported/not-run separately, human hours, conversion overhead and system latency. Rubin remains a dated source-based case study until hardware and release-specific software support are available; advertised peaks are not measured throughput.

**Budget / stop-go.** Proposed 12 hours per available target after qualification. Deliver capability matrix, one reproducible gap/fix and target manifests. Without both targets, finish a source dossier plus one-target experiment, with no cross-vendor performance ranking. A new proprietary target additionally requires its real SDK/runtime contract before optimization work.

<a id="p16"></a>
### P16 — Host-to-device pipeline optimizer

**Hypothesis.** One measured host/input intervention improves the complete model pipeline without excessive buffering. Pick one of NUMA locality, pinned-memory reuse, prefetch depth, dispatch batching or tokenization overlap.

**Execution — 4 weeks.**

1. **Week 1.** trace input preparation, queue wait, copies, launch and completion; validate ordering and buffer lifetimes.
2. **Week 2.** implement the single intervention with explicit backpressure and ownership.
3. **Week 3.** sweep payload size/load with the same model work; inject a slow consumer and a reused buffer to expose queue/lifetime errors.
4. **Week 4.** rerun unprofiled throughput/tails, inspect one confirming profile and measure host/device peak memory.

**Baselines and evidence.** Existing pipeline and a simple synchronous/reference pipeline. Asynchronous APIs do not prove overlap: establish readiness and concurrent progress on the timeline. Include CPU utilization, allocation cost, queue depth and data-quality/output identity. Pinning every input can increase pressure; a prefetch win must survive bounded queues.

**Budget / stop-go.** Proposed 8 device-hours and ordinary CPU tooling. Deliver pipeline source, ownership diagram, raw timeline and a sensitivity report. Stop if the device was never starved and trace a different bottleneck; that is evidence against the original hypothesis. A CPU-only demonstration cannot validate host-to-GPU overlap.

<a id="p17"></a>
### P17 — Adaptive speculative-decoding evaluator

**Hypothesis.** A load-aware speculation policy improves goodput over a fixed draft length while preserving the declared sampling distribution. Acceptance rate alone is not the objective.

**Execution — 6 weeks.**

1. **Week 1.** reproduce categorical accept/reject/residual correction from L10 and test equal/disjoint support.
2. **Week 2.** qualify one real supported draft/target engine path and establish speculation-off and fixed-length baselines.
3. **Week 3.** instrument draft, verification, rollback/cache state, accepted tokens and total iteration cost.
4. **Week 4.** implement a bounded rule choosing off/short/long speculation from recent acceptance and queue pressure.
5. **Week 5.** test low/high acceptance against low/high offered load and adversarial distribution shifts.
6. **Week 6.** freeze policy, test held-out prompt families, evaluate quality and report SLO goodput/latency/memory.

**Baselines and evidence.** [Speculative decoding's original exact-sampling contract](https://arxiv.org/abs/2211.17192), current engine's fixed/adaptive option if available, and no speculation. Compare equal compute reservations including the draft model. Separate greedy from stochastic experiments; exactness in the toy probability test does not certify the whole engine implementation.

**Budget / stop-go.** Proposed 24 device-hours and a fixed evaluation-token budget. Deliver policy, rollback/correctness fixtures, stage samples and loss regimes. Proceed as research only if the policy adds value beyond the current engine's adaptive feature; otherwise reproduce and explain its boundary.

<a id="p18"></a>
### P18 — Domain-transfer deployment study

**Question.** Does an inference optimization validated on LLM kernels remain useful for a different computation? Choose one fixed public vision/classification/segmentation model and one device. The question concerns execution, not creating a new model.

**Execution — 4 weeks.**

1. **Week 1.** freeze the public model, dataset license, input transform and domain-specific output/quality metrics.
2. **Week 2.** reproduce eager and a supported compiler/AOT/vendor baseline with cold/warm boundaries.
3. **Week 3.** apply one already justified change such as partition/fusion/precision, and test adverse image sizes/classes.
4. **Week 4.** compare model quality, memory, latency, compilation and sustained execution on a held-out split.

**Baselines and evidence.** The target's optimized path and unchanged reference. A classification metric does not validate segmentation quality; define the right output semantics before optimizing. If using a public medical dataset, this is a deployment engineering study with no clinical-validation or patient-care claim. Use public/synthetic inputs and avoid transferring private data to optimization services.

**Budget / stop-go.** Proposed 12 target-hours. Deliver dataset/preprocessing manifest, quality evaluator, artifact and a transfer-success/failure report. Stop at one domain/device; if the improvement depends entirely on domain-specific kernels, explain that finding rather than claiming a universal optimizer.

<a id="p19"></a>
### P19 — Reproducible optimization workbench

**Question.** Can two independent experiments share an evidence contract while keeping incompatible SDKs and decision policies separate? A harness is useful infrastructure; novelty requires evidence beyond adding a UI.

**Execution — 4 weeks.**

1. **Week 1.** agree on immutable workload/environment/candidate/result identities with two project users or two existing real experiment scripts.
2. **Week 2.** implement a subprocess adapter with timeout, cancellation, logs and raw artifacts; support CPU first and a target adapter where available.
3. **Week 3.** implement comparison reports with explicit measured/simulated/source-only provenance; reproduce build failure, wrong-result rejection and interrupted execution.
4. **Week 4.** have a clean environment rerun both experiments, independently recompute key metrics and test resuming the archive.

**Baselines and evidence.** Minimal shell/script execution with the same experiments and an eligible existing benchmark schema such as [FlashInfer-Bench](../references/project_research_review.md). Measure setup/rerun time, archive completeness, failure classification and controller overhead; “many records stored” is not a success metric. Preserve rejected candidates and prevent a proposer from changing the evaluator's criteria.

**Budget / stop-go.** CPU development plus at most 8 device-hours for adapter qualification. Deliver schema, CLI, two adapters and rerun report. Defer dashboards, scheduling clusters and Rust rewrites until both evidence paths work. Stop platform generalization if it makes the first two experiments harder to reproduce.

<a id="p20"></a>
### P20 — Rollout/trainer resource-balance study

**Hypothesis.** A fixed-budget allocation rule increases useful accepted rollout/training progress over a static split. Start with a calibrated queue model; a real post-training loop is a separate measured milestone.

**Execution — 6 weeks.**

1. **Week 1.** define generation, scoring/filtering, weight distribution and training stages with model-version identity.
2. **Week 2.** construct stage accounting and validate conservation against synthetic hand-worked cases.
3. **Week 3.** measure one small supported rollout/update loop if resources permit; otherwise publish calibration requirements.
4. **Week 4.** compare static and simple queue-pressure allocation under the same total devices and accepted-data policy.
5. **Week 5.** inject reward/scorer slowdown, long generations and weight-update cache invalidation.
6. **Week 6.** evaluate held-out workloads, learning/quality progression and sensitivity to service-time errors.

**Baselines and evidence.** Static resource split and current supported training/rollout configuration. Record useful/accepted tokens, rejected/stale rollouts, trainer idle time, version lag, transfer cost and total elapsed time. Raw tokens per second are insufficient if useful samples or learning progress worsen. Adaptive allocation and cache reuse are not allowed to consume incompatible weight versions silently.

**Budget / stop-go.** Proposed CPU model first, then 12 cluster-hours on 2–8 GPUs after fit. Deliver stage/identity ledger, allocation policy and measured-versus-modeled report. If a real loop cannot fit, finish a source/analytical capstone and make no post-training speedup claim.

<a id="p21"></a>
### P21 — Search-history transfer and stopping rules

**Research question.** When does retrieving prior optimization trials reduce total search cost on unseen shapes or hardware, and when should the planner abandon that history? Persistent knowledge is already part of [KernelEvolve](../references/project_research_review.md); the proposed contribution is a controlled transfer/stopping study.

**Execution — 8 weeks.**

1. **Week 1.** choose a finite search task from P02/P06/P07 and predeclare a feasible-performance target.
2. **Week 2.** collect successes and failures across at least three source shape families, with environment hashes and full search costs.
3. **Week 3.** create train/validation/test splits by family, reserving one target architecture if available.
4. **Week 4.** implement nearest-neighbor history, hardware-aware retrieval and no-history search.
5. **Week 5.** add one confidence/age-aware fallback-to-exploration rule.
6. **Week 6.** compare five-seed searches with equal wall/device/API budgets and a shuffled-history negative control.
7. **Week 7.** inject compiler/version drift and misleading near-neighbor histories.
8. **Week 8.** rerun held-out trials and analyze the time-to-target and best-feasible-performance frontiers.

**Baselines and evidence.** Existing tuner, random search, simple feature-based ranking and history without the new fallback. Report negative-transfer frequency, failed-trial savings, retrieval overhead and stopping regret against an offline best-of-tested-candidates reference. Never call that finite reference a global optimum. Revalidate a retrieved implementation; history can rank candidates but cannot grant correctness.

**Budget / stop-go.** Proposed 40 device-hours, plus a fixed generation cap if used; archived-data analysis alone cannot establish new-device performance. Deliver a versioned trial dataset, split protocol, retrieval/rule implementation and uncertainty report. Novelty requires a new related-work search on transfer tuning; a failure to beat nearest-neighbor retrieval is useful evidence.

<a id="p22"></a>
### P22 — Adversarial numerical validator for generated kernels

**Research question.** Which combination of differential, metamorphic and boundary tests catches incorrect “fast” candidates at acceptable validation cost? [KernelBench, FlashInfer-Bench and the runtime-share preprint](../references/project_research_review.md) already discuss correctness limitations; this project measures a small validator design rather than claiming universal proof.

**Execution — 6 weeks.**

1. **Week 1.** select reduction, masked softmax and one GEMM contract; reproduce the pinned evaluators' current acceptance behavior.
2. **Week 2.** create labeled mutants: partial write, wrong tail, missing mask, constant output, input mutation and incorrect stride assumption. Include correct low-precision/reordered-reduction implementations.
3. **Week 3.** implement scale-aware error checks, finite/shape checks and operation-specific metamorphic tests where mathematically valid.
4. **Week 4.** measure false acceptance, false rejection and cost on a development set.
5. **Week 5.** freeze thresholds and test hidden mutants, real generated failures and previously unseen magnitudes/shapes.
6. **Week 6.** evaluate the complete gate before timing in an independent process and publish counterexamples.

**Evidence.** Confusion matrix by bug class and magnitude, confidence intervals, validation time and accepted-kernel performance. Use a stable denominator/absolute fallback near a zero reference; relative error alone is also inadequate there. Statistical sampling tests are separate from deterministic tensor tests. Mirage's restricted equivalence guarantees do not certify arbitrary floating-point kernels.

**Budget / stop-go.** Proposed 20 device-hours; no model-generation cost is required for the mutant-based MVP. Deliver mutation fixtures, thresholds, reference implementation and a concise limitations statement. Accept the study if it reveals a reproducible blind spot or improves a declared detection/cost tradeoff without excessive false rejection. Never promote a candidate that fails the frozen contract.

<a id="p23"></a>
### P23 — Cache-operation identity and restoration protocol

**Research question.** Can a narrowly specified cache-state protocol preserve the intended computation across branch, edit, eviction and restoration? [AKV](../references/project_research_review.md) already establishes that drop/reposition history matters. The new hypothesis concerns a limited property/interop boundary, not inventing cache engineering or sharing arbitrary KV tensors across engines.

**Execution — 8 weeks.**

1. **Week 1.** read AKV's operation/state model and pin a supported implementation; specify ordinary exact-prefix behavior separately.
2. **Week 2.** build a CPU reference state machine keyed by weights/adapters, token/position history, cache operations and format/layout where relevant.
3. **Week 3.** generate paired histories with identical final visible tokens but different operations; include tenant separation and shared branches.
4. **Week 4.** define a manifest and conservative compatibility predicate, with unsupported histories rejected.
5. **Week 5.** integrate one engine path if supported, using operation-aware reconstruction as the reference.
6. **Week 6.** test evict/restore, cancellation, partial transfer and stale-version rejection.
7. **Week 7.** run held-out history sequences and one contention workload.
8. **Week 8.** publish correctness counterexamples, compatibility limits and measured lookup/reconstruction overhead.

**Baselines and evidence.** AKV's identity/reuse mechanism, token-only matching as an intentionally unsafe control, and full reference execution under the **same declared operation semantics**. Dropping attended context changes the model computation; separately compare task quality against full context before claiming unchanged user behavior. Equal strings or a session ID do not establish state equivalence.

**Budget / stop-go.** CPU property tests first; proposed 24 target-hours after backend qualification. Deliver operation-history corpus, state-manifest spec and tested compatibility rules. If the backend cannot expose the required operations, complete a protocol/property study with “not run” integration cells. Linear-attention support is an optional future project, not silently assumed.

<a id="p24"></a>
### P24 — Distributed optimization candidate evaluator

**Research question.** Can a reproducible evaluator compare communication/compute-overlap candidates while rejecting invalid collective behavior and counting stragglers? [FlashInfer-Bench v1](../references/project_research_review.md) explicitly excludes multi-GPU/communication kernels; current upstream scope must be checked again before claiming an implementation gap.

**Execution — 8 weeks.**

1. **Week 1.** select two homogeneous devices, one supported NCCL or RCCL communicator and a fixed all-reduce/compute workload.
2. **Week 2.** define rank-local input/output identities, rendezvous, timeout and maximum-rank completion records.
3. **Week 3.** reproduce library defaults and validate collective outputs against an independent reference.
4. **Week 4.** wrap a finite set of bucket sizes or scheduling choices; use existing collectives, not a new transport.
5. **Week 5.** inject delayed rank, missing rank and wrong reduction/division controls with clean process-group teardown.
6. **Week 6.** compare candidates under equal device reservation and fixed work, measuring readiness and exposed wait.
7. **Week 7.** test unseen payload/skew regimes and one training-step integration.
8. **Week 8.** publish schemas, traces and rerun protocol.

**Evidence.** Correct reductions/updates, per-rank and maximum-rank time, synchronization cost, failures, abandoned allocations and end-to-end step gain. A faster launch without completion is invalid. Multi-GPU timing requires a consistent completion boundary; do not subtract unsynchronized host clocks as transfer duration.

**Budget / stop-go.** Proposed 20 paired-device hours, a hard timeout and no unmanaged shared-cluster jobs. Deliver a bounded evaluator and one demonstrated measurement failure it prevents. Stop if communicator/runtime support is unavailable; a CPU Gloo study may validate protocol behavior but cannot establish GPU overlap performance.

<a id="p25"></a>
### P25 — Heterogeneous edge multi-model scheduling

**Research question.** Can a bounded placement/scheduling rule improve deadline completion for two models under sustained memory and thermal constraints? This extends the supplied strategy toward a practical CPU/GPU/NPU board. Use one small language workload plus one public auxiliary model, such as vision or audio, only when both are supported by the device SDK.

**Execution — 8 weeks.**

1. **Week 1.** audit model/backend/operator/precision support and available power/thermal instrumentation.
2. **Week 2.** reproduce each model alone on supported targets; record export, partition and fallback boundaries.
3. **Week 3.** create joint arrival traces and measure interference at fixed quality.
4. **Week 4.** implement a simple bounded-queue scheduler and compare serial, fixed-placement and vendor-default policies.
5. **Week 5.** add one measured rule for placement or admission using backlog and observed service time.
6. **Week 6.** run sustained low/high-load trials with cooldown/reset criteria and logged temperature/frequency if exposed.
7. **Week 7.** test unseen bursts and sensor/service delays; check starvation and quality.
8. **Week 8.** publish device-specific results and a second-target feasibility dossier rather than an untested portability claim.

**Evidence.** Deadline success, task/LLM latency, per-model fairness, total memory and sustained degradation. Measure energy per useful task only with an actual power boundary and sampling method; temperature is not an energy meter. Count CPU fallbacks and transfers. Backend node coverage is insufficient to predict joint performance.

**Budget / stop-go.** Proposed 30 board-hours with scheduled access; no additional device purchase. Deliver workload traces, scheduler, supported-placement matrix and sustained-run report. If only one model/target works, narrow to P14 instead. A publishable policy needs stronger prior-art and generalization evidence than a single-board integration win.

<a id="p26"></a>
### P26 — GPU selection and decoding control

**Question.** When does faster selection materially reduce full decode latency or improve SLO goodput? Keep vocabulary sampling, beam candidate selection and sparse-attention context selection as distinct tracks. The MVP chooses vocabulary sampling; beam search is an optional extension. Read the [bottleneck literature review](../references/inference_bottlenecks_review.md), including Qrita, FlashInfer and GVR, before claiming novelty.

**Execution — 6 weeks.**

1. **Week 1.** Qualify one model/engine/GPU; profile linear layers, attention, output projection, selection and host overhead. Freeze top-k/top-p order, ties and sampling contract.
2. **Week 2.** Build reference fixtures for peaked/flat distributions, duplicates, masks, large vocabularies and shape tails; validate support/distribution or exact filtered output as appropriate.
3. **Week 3.** Compare the installed optimized sampler with full-sort and one eligible selection/pivot implementation under equal tuning budgets.
4. **Week 4.** Test one bounded intervention such as fused filtering or shape-dependent dispatch. Record compilation, temporary bytes, launch count and failed trials.
5. **Week 5.** Measure held-out batches/vocabularies/contexts and full-request latency/goodput. Include a case where sampling is a small fraction and the optimization loses or barely matters.
6. **Week 6.** Reproduce winners and report component versus system gains. An optional beam track adds a tiny exhaustive score/ancestry/EOS reference and measures forward/state-management costs separately.

**Baselines and evidence.** Existing optimized engine path plus FlashInfer/Qrita where supported. Compare the same computation/sampling semantics; changing beam width or truncating context is a separate quality experiment. Use the [diagnosis worksheet](../labs/kit/inference_bottleneck_worksheet.md). Five search seeds apply if tuning is part of the research claim.

**Budget / stop-go.** Proposed 20 device-hours after the fit pilot. Deliver kernel/engine adapters, correctness fixtures, raw phase traces, end-to-end results and a decision map. No GPU is a source/analytical study, not measured acceleration. Reproducing an existing method is valid coursework; a research claim needs a new testable tradeoff beyond a wrapper.

## A5. Suggested routes and release sequence

**Data-center optimization route.** Complete P01 or P19's minimal evaluator, then P06/P02, then a bounded P09. P08 is optional when the profile identifies addressable custom-kernel work. P21/P22 are independent research branches that can improve the evaluator/planner later.

**Agent systems route.** P03 tests exact-state lifecycle scheduling; P23 tests history-dependent state correctness. P04 adds measured placement/handoff only when two devices and connector support exist. Do not combine all three interventions in the first experiment.

**Edge/vendor route.** P14 qualifies one target, P15 diagnoses a concrete software gap, and P25 studies two supported workloads under contention. A proprietary accelerator with no usable compiler/runtime interface requires software bring-up before any optimizer can create value.

**Training route.** P13 is a reliable local start, followed by P12 or P24 on a real allocation; P20 is a later integration study with model-version and learning-progress controls.

For each chosen project use serial reviewable PRs: **baseline and proposal → correctness/evaluator → one intervention → held-out results and report**. Merge and refresh `main` before the next PR under the repository workflow. The appendix lists options; it does not schedule or implement all projects. For a research paper, add a current related-work comparison, artifact/reproduction checks and a precise statement of what the experiments do and do not establish. Use the [capstone rubric](../labs/kit/capstone/CAPSTONE.md) for assessment; a defensible null result can earn full credit.
