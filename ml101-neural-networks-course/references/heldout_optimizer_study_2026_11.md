# Validation-selected optimizer learning study — 2026-11-12 UTC

[Deck 08 slide 10](../slides/08_optimizers.html#slide-10) · [L2 runner](../labs/L2_optimizers/heldout_study.py) · [protocol](../labs/L2_optimizers/heldout_protocol.json) · [all-trial search artifact](../labs/L2_optimizers/heldout_search_report.json) · [held-out artifact](../labs/L2_optimizers/heldout_test_report.json).

## Frozen question and sequence

This is an executed synthetic linear-predictor learning study, extending FND-2 beyond the earlier deterministic-quadratic/protocol exercise. It is not a universal optimizer ranking, deep-network experiment or student pacing pilot. The [primary Tuning Playbook reading](https://github.com/google-research/tuning_playbook/blob/7554a51f5d3002ba90308d39701b586f5a9b962e/README.md#designing-the-next-round-of-experiments) distinguishes scientific, nuisance and fixed hyperparameters; the protocol and allocation below are course choices, not a reproduction of a paper benchmark.

Protocol commit `64c436c` preceded execution; implementation/controls commit `624ed66` preceded search; search/selection commit `70c1d0c` preceded the single held-out evaluation invocation. These are topic-branch provenance commits (a later squash may not retain them in main). Canonical JSON protocol/selection digests and executed source byte hashes are also in the artifacts. The held-out artifact identifies the exact search artifact it consumed.

- Existing `optimizers.noisy_regression`: 384 examples, eight features, noise 0.5, pool seed 101. One teacher coefficient vector is shared by the entire pool. A permutation with seed 102 assigns train/validation/test IDs 192/96/96. IDs and exact feature rows are checked for cross-split duplication. No preprocessing fits on validation/test; this fixture uses none.
- Existing SGD and Adam implementations and L0 MSE/gradient are reused unchanged. Scientific variable: optimizer. Nuisance variable: LR. Frozen grids: SGD [.003, .03, .3], Adam [.001, .01, .1]. Each gets three candidates × seeds [11,23,37] × 300 updates, batch 16 with replacement, float64 and constant LR. Adam beta1 .9, beta2 .999, epsilon 1e-8, zero decay. The seed-level initialization and batch streams match across all candidates/optimizers; their digests are recorded.
- Each final checkpoint is scored after the last update, rather than using the existing optimization helper's pre-update history. Select one LR per optimizer by mean validation MSE across all three seeds; a failed seed invalidates its candidate; exact ties use grid order. No adaptive extension or test-driven checkpoint selection.
- Search stores zero test metric calls. Its locked selections/checkpoints were committed before evaluation. The second stage performs six held-out scores (three fixed-seed models per selected LR), with zero additional fitting. It refuses changed protocol/source/selection/checkpoint identities and the CLI refuses overwriting an existing output. The same known synthetic test split is regenerable in Python: this is process separation, not secure or blinded data isolation.

## All candidate means and selected held-out outcomes

| Optimizer | LR | Mean validation MSE | Selected? |
|---|---:|---:|---|
| SGD | 0.003 | 0.51043435 | No |
| SGD | 0.03 | 0.25786022 | Yes |
| SGD | 0.3 | 0.40896083 | No |
| ADAM | 0.001 | 6.01445755 | No |
| ADAM | 0.01 | 0.29619734 | Yes |
| ADAM | 0.1 | 0.31703701 | No |

| Optimizer | Seed 11 test MSE | Seed 23 | Seed 37 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| SGD | 0.31343169 | 0.31348560 | 0.28979068 | 0.30556932 | 0.01366473 |
| ADAM | 0.30367666 | 0.30481647 | 0.30407008 | 0.30418774 | 0.00057894 |

The paired Adam-minus-SGD difference changes sign across seeds. These three optimization seeds reuse one small fixed split, so their SD is not dataset-level uncertainty or a confidence interval. The limited grids, fixed hyperparameters, final-only checkpoint rule and equal-update budget condition every conclusion. Adam .001's validation MSE 6.01446 is an adverse underfit candidate retained in the full ledger; no main-study trial failed and no failures are invented. The controlled overflow test is a separate fixture, not a main-study result.

## Executed cost and environment

Both optimizers completed all nine planned fits: **2,700 updates each, 5,400 total**. There were 18 finite search records and six selected test records; no dropped seed, final refit, extra hyperparameter search or held-out retraining. Per-trial fit/validation intervals and aggregate search/test intervals are stored as local wall-clock observations, not portable benchmarks. Fit intervals include training, scoring and checkpoint serialization/hash work, but exclude initialization/batch preparation; the search interval includes generation/setup/selection/hashing. Imports, CLI parsing and report-file writing are outside those intervals. Equal updates are not equal arithmetic cost, memory or runtime.

The report records CPython 3.12.14 / NumPy 2.3.5 on the authoring CPU, actual platform string and source/data/stream/weight hashes. No package installation or dependency-pin upgrade was needed. Re-running this now-known study checks reproducibility; it does not create a fresh independent held-out result. Future tuning after inspecting these results must treat this test as development evidence and use a new evaluation cohort.

## Controls and validation

Four new test methods plus twelve existing L2 tests pass (16 total). Controls include a literal one-step derivative/final-weight oracle; mean-versus-luckiest-seed selection and training-loss decoy; failed-candidate retention without reallocated budget; missing/duplicate trial rejection; overlapping ID and exact-feature leakage rejection; altered held-out labels leaving search weights/selections unchanged; independent held-out squared-error reduction; and changed selection/checkpoint rejection. None checks a hard-coded SGD/Adam winner. Original optimizer, model/gradient and historical result tables remain unchanged.

This completes the bounded FND-2 reference learning experiment alongside its earlier teaching/protocol work. A broader dataset/model study and classroom pilot remain separate; other audit items, including AGT-2's measured memory benchmark beyond PR #120, are still open.
