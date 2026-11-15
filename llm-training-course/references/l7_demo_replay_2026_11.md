# L7 unchanged demo replay — 2026-11-15 UTC

**Six fresh-process demos completed**, three seeds in each of two existing CPU environments;18 printed stage rows are retained. Every printed value agrees across these two current cohorts, while the seed0 GRPO/Dr. GRPO results differ from the historical example. This bounded TRN-4 replay does not establish held-out arithmetic quality, exact historical provenance, bitwise equivalence, or an algorithm ranking.

## Primary sources and predeclared scope

[Protocol](../labs/L7_grpo/demo_replay_protocol.json), wrapper and three result-boundary controls were committed at **9afcec2 before training** (base main `2f3490a`). The wrapper calls the unchanged `grpo.main()` with its default300-step demo and seeds0/1/2; source hashes pin `grpo.py`, `policy.py` and `task.py`. It changes no algorithm, reward, mask, data split, checkpoint-selection rule or historical table. All methods/seeds run once, no retries/tuning/dropped outcomes. The wrapper captures only the demo's existing stdout: rewards at3 decimals, conditional response lengths at2 decimals. It does not recover unprinted full-precision metrics, raw responses or model checkpoints.

[Liu et al. v1, §3.1–3.2](https://arxiv.org/html/2503.20783v1) motivates removing response-length and group-standard-deviation normalizations. The paper evaluates different models/tasks/training settings. Here the original small course implementation uses a fixed `MAX_LEN` denominator and a digit/filler task; we neither import its author framework nor reproduce the paper's benchmark. [PyTorch2.15 reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html) cautions that seeding does not guarantee cross-release/platform replay. Equal printed values below do not imply equal tensors or trajectories.

## Exact course path and evaluation population

For each worker, `main()` seeds Torch, chooses30 SFT prompts with `split(seed)`, trains the existing `Policy` for300 SFT steps, evaluates, and clones that SFT state in memory for separate GRPO and Dr. GRPO300-step runs. RL defaults remain16 prompts ×G8 responses per step,2 inner updates, Adam LR0.003, no KL. Each arm resets Torch seed before policy construction/loading, uses the same in-memory SFT state, and has its own seeded prompt/sampling generators. This is the original demo's pairing, not a newly instrumented state-hash qualification.

**RL and all three evaluation calls use the same full100 ordered digit-pair prompt pool.** Every worker records the full RL/evaluation list and its30 SFT prompts, all contained in that pool. `evaluate()` uses8 stochastic responses per prompt, its default seed0 for every stage/worker, and the existing exact-answer verifier/mask; reward averages800 responses. Correct/wrong length averages are conditional on different outcome subsets, include the first EOS and exclude post-EOS padding. They are not a matched per-response length treatment effect. No evaluation holdout, semantic/human judge, new scoring or raw response trace is added. The separate coverage/steering experiments keep their own frozen results and holdouts.

## Environments and all printed outcomes

Existing CPython3.12.14 /macOS27.0.1 arm64 CPU cohorts: `torch291` =Torch2.9.1/NumPy2.3.5 from the [tested CPU lock](tested_cpu_lock_2026_11.md); `torch215` =Torch2.15.0/NumPy2.5.4 from the [CMP-1 environment](../../llm-compute-systems-course/references/executorch_cpu_compatibility_2026_11.md). One intra-op and one inter-op thread; deterministic algorithms disabled; original float32 CPU policy defaults. Full package/Torch build records are in each worker. No new installation/download or default-pin migration. Cohort/build and unrecorded original environment details prevent attributing historical differences to one version.

Every table triple is **mean reward / mean correct-response length / mean wrong-response length**, at the demo's original printed precision. No averaged winner or selected seed is substituted.

| Cohort | Seed | SFT: reward / correct length / wrong length | GRPO after300 | Dr. GRPO after300 |
| --- | --- | --- | --- | --- |
| torch291 | 0 | 0.273 / 3.93 / 4.80 | 0.498 / 2.43 / 2.82 | 0.484 / 2.44 / 2.59 |
| torch291 | 1 | 0.282 / 3.83 / 4.13 | 0.419 / 4.48 / 4.46 | 0.416 / 4.27 / 4.44 |
| torch291 | 2 | 0.270 / 3.91 / 4.33 | 0.407 / 3.98 / 3.99 | 0.389 / 2.49 / 2.47 |
| torch215 | 0 | 0.273 / 3.93 / 4.80 | 0.498 / 2.43 / 2.82 | 0.484 / 2.44 / 2.59 |
| torch215 | 1 | 0.282 / 3.83 / 4.13 | 0.419 / 4.48 / 4.46 | 0.416 / 4.27 / 4.44 |
| torch215 | 2 | 0.270 / 3.91 / 4.33 | 0.407 / 3.98 / 3.99 | 0.389 / 2.49 / 2.47 |


Historical seed0 triples remain: SFT0.273/3.93/4.80, GRPO0.506/2.43/2.79, Dr. GRPO0.489/2.46/2.49. Both current cohorts reproduce the SFT row at printed precision but not the complete two RL rows. Current seed0 reward differences from the printed historical values are−0.008 and−0.005. These are rounded descriptive differences, not full-precision error bounds or significance tests. Both methods' current rewards improve over their corresponding SFT printout; that observation concerns this training/evaluation pool, not held-out transfer.

Do not overwrite the historical values, infer which original build produced them, or claim the replay authenticates historical provenance. Source/build/thread/raw-output omissions in the older record remain unresolved. In particular, equal current printed results across releases do not prove identical random draws, trained weights, gradients, trajectories or unseen-task behavior. The original paper's algorithm conclusions and this toy demonstration have distinct evidence scopes.

## Validation and work accounting

All six workers /18 stdout rows complete, with no missing, failed, refused or retried run. Each cohort passes **31 L7 tests** (the existing28 plus3 new result-boundary controls). The three new tests also pass before training with standard-library Python3.9.6: reject missing/extra stage output, wrong order or invalid ranges, missing/duplicate workers, and altered parsed metrics. A retained metadata check verifies source hashes, thread settings, full shared RL/eval pool, SFT membership and all six workers; this is a reconstruction aid, not authenticated runtime evidence.

The fixed unchanged loops imply:6×300 =1,800 SFT optimizer updates on57,600 sampled examples;12 RL arms ×300 =3,600 rollout batches and7,200 inner optimizer updates;460,800 sampled RL responses /3,686,400 executed sampler slots;18 evaluation stages ×800 =14,400 sampled evaluation responses /115,200 slots. Slots include post-EOS sampler positions; valid-token counts and unique discoveries are not measured. These are **source-derived counts for completed fixed loops**, not newly instrumented per-step hardware counters or equal-FLOP claims. The complete test suites execute additional work outside those demo counts.

Summed intervals around `grpo.main()` are **10.464s**, including SFT, both RL arms and all three evaluations. The sequential driver interval is16.981s, including six interpreter startups and both complete L7 test suites; no concurrent demos. Earlier protocol tests, collector/reporting and prior environment setup/downloads are outside that driver interval. Peak worker process RSS is277,299,200–306,839,552 bytes (macOS units), including imports/model/run; not GPU/per-operator memory or portable throughput.

[Summary](../labs/L7_grpo/demo_replay_artifacts/summary.json), [verification](../labs/L7_grpo/demo_replay_artifacts/verification.json), [driver timing](../labs/L7_grpo/demo_replay_artifacts/driver_timing.json), six worker JSON/stdout logs and the [2.9.1](../labs/L7_grpo/demo_replay_artifacts/torch291-tests.log)/[2.15.0](../labs/L7_grpo/demo_replay_artifacts/torch215-tests.log) test logs retain the complete observation. The collector reparses all original stdout, checks the full planned worker set, and records worker-file hashes.

From `labs/L7_grpo/`, use a matching cohort and fresh output directory:

```sh
python -m unittest -v
python demo_replay.py worker --lane torch291 --seed 0 --output /tmp/demo-new/torch291-s0.json
# Repeat seeds0/1/2 for torch291 and torch215 with their matching interpreters.
python demo_replay.py collect --directory /tmp/demo-new --output /tmp/demo-summary-new.json
```

Outputs are exclusive; changed original source, wrong package cohort or altered/missing output is rejected. Keep version/build/thread evidence with a rerun rather than treating a broad Torch2 prerequisite as a reproduction recipe.

**Completed:** bounded replay/reporting of the original L7 default300-step demo in two recorded CPU cohorts. **Still open:** exact historical build/provenance, full-precision/raw-response/state replay, held-out task transfer, larger seed studies and real-backend/distributed qualification. No full TRN-4 recertification or paper replication. AGT-2's broader benchmark remains open; PR#120 is teaching/protocol only. Deck28 slide5 now pairs the objective mechanism with these results, retains the prior paper explanation in hidden notes and has [before/after captures](../../docs/reviews/training-l7-demo-replay/README.md). There is no translated deck folder in this course.
