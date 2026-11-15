# L1 600-step training-table replay — 2026-11-15 UTC

**24 fresh-process training runs completed** with the existing unchanged L1 implementation. Torch2.9.1 matches all four historical rows at their printed three-decimal mean/SD precision; Torch2.15.0 matches three. The new constant-rate mean is1.732790108 (prints1.733), compared with historical1.734. This is a bounded TRN-4 reproducibility observation, not a corrected historical provenance claim, optimizer ranking, or complete historical-table replication.

## Frozen scope and primary reading

Base main `a258d1c`. [Protocol](../labs/L1_pretraining_loop/table_replay_protocol.json), wrapper and four collector controls were committed at **16b09141befc16cd46e9c2cdcbce819c2099ddb5 before training**. It fixes the four existing default recipes (AdamW constant/cosine/WSD and Muon-WSD), seeds0/1/2,600 steps, peak LR0.003, decay0.1 for hidden weights,5% warmup and unchanged WSD defaults. All recipes and final-step results are reported; no search, winner selection, retries or omitted outcomes. The historical1,500-step and high-LR/QK/z-loss tables, L7 training table, other labs, GPUs and distributed execution are outside this pass.

[PyTorch2.9 reproducibility notes](https://docs.pytorch.org/docs/2.9/notes/randomness.html) and [2.15 notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html), checked on this date, distinguish RNG seeding from deterministic execution and warn against assuming replay across releases/platforms/devices. Our observations do not contradict that boundary: agreement at rounded table precision is weaker than bitwise identity. No blanket determinism or speed claim is inferred from those documents.

The wrapper calls **the existing `train.train()`**, without editing `train.py`, `model.py`, `optim.py` or `schedules.py`; their frozen hashes are in every worker and the protocol. Those course files are the primary implementation under test. Model:2 layers, width64,4 heads, context32,12-character vocabulary, CPU float32, public PyTorch SDPA with its existing default dispatch. No compile/AMP/backend override or algorithm reimplementation. Four intra-op threads and one inter-op thread per process; no CPU affinity/core reservation. This does not recreate the original table's unspecified four-core machine.

## Existing environments and data

Reused the [TRN-4 Torch2.9.1/NumPy2.3.5 tested CPU cohort](tested_cpu_lock_2026_11.md) and the [CMP-1 Torch2.15.0/NumPy2.5.4 candidate cohort](../../llm-compute-systems-course/references/executorch_cpu_compatibility_2026_11.md), both CPython3.12.14 on macOS27.0.1 arm64. No package installation/download, default-pin migration or new hash lock. Every worker records all installed package versions, Torch build configuration and thread/determinism settings. Different transitive cohorts and uninstrumented dispatch mean this is not a clean causal attribution to one library version.

`train()` resets Torch seed per run, and creates its own batch generator from that seed. Data generation retains fixed Python local RNG seeds1/2 and validation evaluation uses a fresh generator seeded0 at every logged interval. Deterministic algorithms remain disabled. Same seed declarations do not establish identical initial weights, batch streams or kernels across versions; those tensors were not separately hashed in this pass.

All24 workers have identical source/data content hashes. Train data:20,000 addition lines /245,456 characters; validation:2,000 lines /24,543 characters. SHA-256 train `bccddf3ac2e88a6f7b1b679fb90fa1d08df79e3383385fdc729a86647ccdc5f4`; validation `541973d6057a5509560009dca6864472a06019df0fbb8b62e9fbd3bfdf73ef07`. **43 unique exact generated lines overlap**; seeds do not make this a disjoint holdout. We preserve this known property to replay the existing experiment, rather than silently repair its data. Loss is mean next-character cross-entropy on sampled validation windows, not a strict addition-answer success rate, clean held-out generalization or language-model BPB.

## Observed table and all seeds

| Recipe | Historical 600-step mean ± SD | Torch2.9.1 replay | Torch2.15.0 replay |
| --- | --- | --- | --- |
| adamw-constant | 1.734 ± 0.018 | 1.734 ± 0.018 | 1.733 ± 0.018 |
| adamw-cosine | 1.748 ± 0.010 | 1.748 ± 0.010 | 1.748 ± 0.010 |
| adamw-wsd | 1.726 ± 0.015 | 1.726 ± 0.015 | 1.726 ± 0.015 |
| muon-wsd | 1.716 ± 0.006 | 1.716 ± 0.006 | 1.716 ± 0.006 |


Mean and **sample** SD across all three seeds, rounded only for display. Historical entries have only printed precision; the replay retains full-precision final losses and twelve curve points per run. Torch2.15 constant-rate mean differs from historical rounded mean by−0.001209892; its mean differs from this Torch2.9.1 lane by−0.001018122. This small difference is descriptive, not a significance test or numerical failure. Three seeds on shared data do not justify a universal ranking.

| Lane | Recipe | Seed0 | Seed1 | Seed2 |
| --- | --- | --- | --- | --- |
| torch291 | adamw-constant | 1.713056684 | 1.741811603 | 1.746556401 |
| torch291 | adamw-cosine | 1.738432139 | 1.757537171 | 1.747387633 |
| torch291 | adamw-wsd | 1.709116891 | 1.737632930 | 1.731362581 |
| torch291 | muon-wsd | 1.721549451 | 1.709310278 | 1.717085928 |
| torch215 | adamw-constant | 1.712478772 | 1.741886795 | 1.744004756 |
| torch215 | adamw-cosine | 1.738432080 | 1.757537290 | 1.747387543 |
| torch215 | adamw-wsd | 1.709155560 | 1.736389652 | 1.732317358 |
| torch215 | muon-wsd | 1.721549138 | 1.709310293 | 1.717092410 |


The Torch2.9.1 lane's4/4 rounded agreement **does not prove the historical table was produced by that version**. Conversely, the Torch2.15 lane's3/4 agreement does not falsify the original2.15 label: exact original build/platform, raw per-seed curves and tensor identities are unavailable in the historical record. Keep the old table dated as historical and attach the new complete observation separately. No historical numbers or labels were overwritten.

## Work and validation

24 workers ×600 steps = **14,400 training updates**, using batch32 × context32 =14,745,600 sampled next-token training positions. Twelve validation checks per run ×8 batches ×64 windows ×32 positions =4,718,592 validation positions. These include repetition and overlap, not unique tokens. Existing tests execute extra training cases; their work is outside these experiment counts. No model/checkpoint search or additional test-set evaluation. The original function returns curves/final loss; final model/optimizer checkpoints were not saved, so no checkpoint/resume or bitwise model-state claim follows.

Summed intervals inside `train()` are **73.535s**, including data/model/optimizer creation, training and scheduled validation. The sequential driver's wall interval is **93.452s**, including both L1 test suites and24 interpreter startups, with no concurrent training workers. Earlier protocol tests, collector/replay analysis, setup and previous installation/downloads are outside that wall interval. Peak process RSS ranges288,309,248–329,400,320 bytes (macOS units), including imports/model/run; not device memory or a per-operator measurement. Timing is one ordered local observation, not a portable speed comparison.

Each cohort passes **14 L1 tests** (10 existing plus4 new collector-boundary controls). The four new controls also passed before training on system Python3.9.6: sample-SD aggregation, missing/duplicate workers, missing/nonfinite curve points, mismatched final values and unplanned seeds. There are24 finite final results and288 finite logged curve points; zero missing/failed runs. No unexpected runtime error or retry occurred.

[Summary and raw worker identities](../labs/L1_pretraining_loop/table_replay_artifacts/summary.json), [independent stdlib arithmetic/identity verification](../labs/L1_pretraining_loop/table_replay_artifacts/verification.json), [driver timing](../labs/L1_pretraining_loop/table_replay_artifacts/driver_timing.json), and per-worker JSON/logs in that directory retain every result. Tests: [2.9.1](../labs/L1_pretraining_loop/table_replay_artifacts/torch291-tests.log) / [2.15.0](../labs/L1_pretraining_loop/table_replay_artifacts/torch215-tests.log). Independent arithmetic recomputes mean/sample SD with `math.fsum` and checks source/data equality, completeness, configured thread counts and source revision; it does not retrain or authenticate execution.

## Replay

Use a matching existing environment for each lane and a fresh output directory. From `labs/L1_pretraining_loop/`:

```sh
python -m unittest -v
# Repeat for every protocol run and seed in each matching lane/interpreter:
python table_replay.py worker --lane torch291 --run adamw-constant --seed 0 --output /tmp/replay-new/torch291-adamw-constant-s0.json
# After all24 worker files exist, collect refuses missing/duplicate/unplanned/nonfinite results:
python table_replay.py collect --directory /tmp/replay-new --output /tmp/summary-new.json
```

The wrapper refuses existing outputs and changed training source hashes. Its worker version guard requires the declared Torch/NumPy cohort. Environment setup is documented in the linked earlier qualification notes; it is not silently supplied by a broad `pip install torch` instruction. Data/build/platform differences need a separately labeled replay.

**Completed:** bounded execution and full reporting of the historical600-step recipe subset in two recorded CPU cohorts. **Still open:** exact historical provenance/bitwise replay,1,500-step/high-LR/L7 tables, clean held-out learning, cross-platform/GPU/distributed qualification and default migration decisions. TRN-4 is not globally recertified; AGT-2's broader benchmark remains open and PR#120 remains teaching/protocol only. Deck01 slide10 shows the result, keeps original closing questions in notes, and has [before/after captures](../../docs/reviews/training-l1-table-replay/README.md). This course has no translated deck folder.
