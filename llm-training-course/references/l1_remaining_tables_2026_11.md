# L1 remaining reference tables — 2026-11-15 UTC

**36 fresh-process fits complete the remaining1,500-step and high-learning-rate reference grids** in two existing CPU cohorts. Every final/curve value is finite; each cohort matches only1/6 historical rows at printed mean/sample-SD precision. Historical numbers remain intact. This is a bounded TRN-4 execution/reporting result, not exact historical reproduction, an optimizer ranking or proof of learning-rate robustness.

Together with the [earlier600-step replay](l1_table_replay_2026_11.md), all10 original L1 reference-table cells now have separately dated observations in both CPU cohorts:60 current fits in total. Execution coverage is not agreement with every historical value or qualification of other platforms/backends.

## Frozen design and primary reading

Base main `5ae9964`. Runner/protocol/control code was committed at `50d9d4b`, with the six-cell selection wording clarified at **95e9ddff46c1d45cc0e8d25d7098290c972d8a8f before training**. [Frozen protocol](../labs/L1_pretraining_loop/remaining_table_protocol.json) declares six preset cells ×seeds0/1/2 ×two existing CPU cohorts. All final-step results are retained with no tuning, winner/checkpoint selection, refit, retry or omitted failed setting. The earlier600-step artifacts/protocol are unchanged.

Four cells use1,500 steps /LR0.003: AdamW constant, cosine, WSD and Muon-WSD. Two use600 steps /LR0.03: AdamW constant without the stability switches and AdamW constant with **both functional RMS QK normalization and z-loss coefficient1e-4**. The current `train.py`, `model.py`, `optim.py` and `schedules.py` are unchanged, source-hash guarded and called directly. Each fit starts afresh;1,500-step runs do not resume the earlier600-step checkpoint. The5% warm-up becomes75 steps for1,500,30 for600; schedules use their full preset budget. Other defaults remain as in the [preceding contract note](l1_table_replay_2026_11.md).

[Wortsman et al. v1](https://arxiv.org/html/2309.14322v1), §§2.2,3.1.1–3.1.2, distinguishes learning-rate sensitivity across a range from one rate and studies QK-layernorm and z-loss with different models/data. Our course uses RMS QK normalization and combines the two switches in one high-rate cell. No isolated QK/z causal attribution, broad LR-sensitivity curve, attention/output-logit trace, paper benchmark reproduction or extrapolation to large models follows. [PyTorch2.15 reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html) bound cross-release/platform replay; seeding is not a promise of bitwise agreement.

## Environment, metric and data boundaries

Existing CPython3.12.14/macOS27.0.1 arm64 cohorts: Torch2.9.1/NumPy2.3.5 ([tested CPU lane](tested_cpu_lock_2026_11.md)) and Torch2.15.0/NumPy2.5.4 ([CMP-1 lane](../../llm-compute-systems-course/references/executorch_cpu_compatibility_2026_11.md)). CPU float32,4 intra-op /1 inter-op thread; deterministic algorithms disabled, original public SDPA dispatch. All package/build/thread records are retained per worker. No new installation/download/default-pin change. Different cohorts and uninstrumented dispatch prevent isolating a library-version effect; neither cohort reconstructs the undocumented original machine/build/code provenance.

Every worker has the same current source/data hashes and the known43 unique exact train/validation line overlaps. Generation remains20,000/2,000 addition lines from seeds1/2, with random next-character windows; see the preceding note for full content hashes. Evaluation averages8 batches ×64 windows ×32 next-token positions with its fixed generator seed0 at each logged interval. Reported validation values are **cross-entropy without the training z-loss term**, even for the combined-switch cell. This is not strict addition-answer accuracy, a clean unseen-task holdout or BPB comparison. Initial weights/batch tensors and final checkpoints were not separately saved/hashed, so identical seed declarations do not establish state-level cross-release replay.

## Complete outcomes

All entries below show mean and sample SD across all three seeds, rounded only for presentation; full-precision finals and every50-step curve point remain in the raw workers. The six cells were fixed before observation.

| Case (run / steps / peak LR) | Historical mean ± SD | Torch2.9.1 | Torch2.15.0 |
| --- | --- | --- | --- |
| adamw-constant / 1500 / 0.003 | 1.587 ± 0.114 | 1.589 ± 0.118 | 1.594 ± 0.109 |
| adamw-cosine / 1500 / 0.003 | 1.655 ± 0.013 | 1.654 ± 0.012 | 1.654 ± 0.012 |
| adamw-wsd / 1500 / 0.003 | 1.575 ± 0.102 | 1.577 ± 0.100 | 1.578 ± 0.097 |
| muon-wsd / 1500 / 0.003 | 1.538 ± 0.139 | 1.538 ± 0.139 | 1.538 ± 0.139 |
| adamw-constant / 600 / 0.03 | 1.758 ± 0.009 | 1.742 ± 0.011 | 1.753 ± 0.011 |
| adamw-constant-qk-z / 600 / 0.03 | 1.717 ± 0.005 | 1.707 ± 0.011 | 1.719 ± 0.003 |


Only the1,500-step Muon-WSD row agrees with its historical printed mean/SD in both cohorts. In the high-rate pair, mean loss drops1.742→1.707 (2.9.1) and1.753→1.719 (2.15) at printed precision. These are observations for the **combined** switches at one preset high rate; they do not separate switch effects or establish a stable LR range. Both high-rate recipes finish with finite logged values; no hidden logit/norm diagnostic was collected to explain a mechanism. Three shared-data seeds do not establish a universal optimizer ranking or significance.

| Cohort | Case | Seed0 | Seed1 | Seed2 |
| --- | --- | --- | --- | --- |
| torch291 | long-adamw-constant | 1.650273755 | 1.453044876 | 1.662831739 |
| torch291 | long-adamw-cosine | 1.640588552 | 1.661199927 | 1.659963757 |
| torch291 | long-adamw-wsd | 1.626238897 | 1.461730883 | 1.642871320 |
| torch291 | long-muon-wsd | 1.614953026 | 1.377061188 | 1.621761039 |
| torch291 | high-adamw-constant | 1.731032744 | 1.742477909 | 1.753815323 |
| torch291 | high-adamw-constant-qk-z | 1.707787648 | 1.717753947 | 1.696514115 |
| torch215 | long-adamw-constant | 1.649658009 | 1.468649760 | 1.664236277 |
| torch215 | long-adamw-cosine | 1.640220538 | 1.661416188 | 1.661393821 |
| torch215 | long-adamw-wsd | 1.626481041 | 1.466563508 | 1.641254902 |
| torch215 | long-muon-wsd | 1.615002543 | 1.377264455 | 1.622199878 |
| torch215 | high-adamw-constant | 1.740988761 | 1.753254697 | 1.763970435 |
| torch215 | high-adamw-constant-qk-z | 1.718533114 | 1.722146288 | 1.716296628 |


Historical labels/numbers stay unchanged. New rounded disagreements do not prove which historical build/code ran, or that a current implementation is defective. Exact original source/build/platform, raw per-seed/trajectory records and tensor identities are absent from the historical example. Report the mismatch as evidence instead of tuning a match or silently replacing the table. This supplement makes no new comparison against a test set or selection on validation.

## Validation and work

36/36 planned workers complete with43,200 training updates and864 finite logged curve points. Each CPU cohort passes **18 L1 tests** (10 existing,4 earlier replay controls,4 new mixed-budget controls). Before training, the new four controls pass in standard-library Python3.9.6: complete sample-SD aggregation, missing/duplicate workers, missing/nonfinite points across the mixed600/1,500 budgets, and final-value/unplanned-seed rejection. Metadata verification checks each preset case, original source hashes, data equality, configured threads and frozen source revision. Independent `math.fsum` arithmetic recomputes the means and sample SD. This is a reconstruction aid, not authenticated execution.

Source-derived fixed-loop work is44,236,800 sampled next-token training positions (43,200×32×32) and14,155,776 validation positions (864×8×64×32); repeated/overlapping positions are not unique tokens. Complete test suites execute additional training outside those experiment counts. Summed `train()` intervals are **211.519s**, including construction/training/scheduled validation; the sequential driver interval is **237.556s**, including36 interpreter startups and both full L1 suites. Prior setup/downloads, earlier protocol tests and later collector/report work are outside that wall interval. No concurrent training workers. Peak worker process RSS spans298,139,648–329,383,936 bytes (macOS units), including imports/model/run—not isolated operator/device memory. These are local observations, not a portable speed comparison.

[Summary/worker identities](../labs/L1_pretraining_loop/remaining_table_artifacts/summary.json), [independent arithmetic/metadata verification](../labs/L1_pretraining_loop/remaining_table_artifacts/verification.json), [driver timing](../labs/L1_pretraining_loop/remaining_table_artifacts/driver_timing.json),36 worker JSON/log pairs and [2.9.1](../labs/L1_pretraining_loop/remaining_table_artifacts/torch291-tests.log)/[2.15](../labs/L1_pretraining_loop/remaining_table_artifacts/torch215-tests.log) test logs are retained. No failure, retry or unexpected runtime warning was observed. Original `train()` returns only final loss/curves, so checkpoint/resume equivalence is not tested.

From `labs/L1_pretraining_loop/`, with a matching lane and fresh output directory:

```sh
python -m unittest -v
python remaining_table_replay.py worker --lane torch291 --run long-adamw-constant --seed 0 --output /tmp/remain-new/torch291-long-adamw-constant-s0.json
# Repeat every protocol case/seed for both matching lane interpreters.
python remaining_table_replay.py collect --directory /tmp/remain-new --output /tmp/remain-summary-new.json
```

Changed current training sources, wrong package cohorts, existing outputs and missing/duplicate/unplanned/nonfinite results reject. The full protocol supplies all six IDs/budgets; preserve raw failures if a different environment produces one.

**Completed:** current two-cohort execution/reporting coverage of all remaining L1 printed reference cells; the earlier600-step and [L7 demo](l7_demo_replay_2026_11.md) remain separately dated. **Still open:** exact historical provenance/state replay, clean held-out transfer, isolated interventions/LR-range studies and cross-platform/GPU/distributed qualification. No whole-course default-pin recertification or broader AGT-2 benchmark completion; PR#120 remains teaching/protocol only. Deck03 slide10 shows the result, retains earlier stability questions in notes and has [before/after captures](../../docs/reviews/training-l1-remaining-tables/README.md). This course has no translated twin folder.
