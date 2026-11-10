# CPU environment observations and configuration reporting

Captured 2026-11-09 for TRN-4. The [machine-readable manifest](cpu_lane_manifest_2026_11.json) records exact versions, source hashes, commands, test IDs, warnings and outcomes for **two separate existing environments**. Neither is a newly resolved installation or a course-wide dependency lock. No lab implementation, existing requirement or historical result table changed.

## What actually ran

| Lane | Observed environment | Executed check | Result and boundary |
|---|---|---|---|
| L1 | CPython 3.12.14, PyTorch 2.9.1, NumPy absent; Darwin arm64, CPU float32, one PyTorch thread | Existing `test_pretraining` suite | 10 passed, no skips/errors. Includes AdamW/Muon/schedule/model checks and two 120-step training assertions. Not the historical 600/1,500-step table on PyTorch 2.15. |
| L9 | CPython 3.12.14, NumPy 2.3.5, PyTorch absent; Darwin arm64, Accelerate BLAS (reported version unknown) | Existing `test_eval` suite | 8 passed, no skips/errors: pass@k, ratings and contamination-estimator checks. Not a rerun of the historical NumPy 2.5 demo tables. |

Both lanes set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. L1's PyTorch thread count was observed as 1. Those environment variables do not establish that every BLAS implementation uses one thread. L1 did not enable deterministic algorithms, and its import emitted the recorded warning that NumPy was absent. The executed suite does not use Torch-to-NumPy conversion. The two package sets must not be presented as a jointly tested PyTorch/NumPy environment.

The executed source revision is `01bec479d4aa5d682304cf53b0414fc73a4982f3`. Every `.py` in each selected lab is hashed there. The wrapper used the standard unittest default loader and verbose runner, with the lab directory on the import path as for `python -m unittest`; it did not replace assertions or override test RNG seeds. The manifest's elapsed test-run times are not training-performance benchmarks.

## Data and split evidence from the executed L1 contract

The training assertion predeclares `adamw-wsd` and `muon-wsd`, seed 0, 120 steps each, learning rate 0.003, weight decay 0.1 and 5% warm-up. Both execute; no winner or checkpoint is selected. The check inspects final-step validation loss, not an independent model-quality test.

Training text uses 20,000 generated additions with Python generator seed 1; validation uses 2,000 with seed 2. The manifest stores both full UTF-8 text hashes. Splitting the generated text on `;` yields 19,777 unique training examples, 1,998 unique validation examples and **43 shared exact examples**. This is an example-string check, not a token-window or semantic contamination rate. It counts literal generated strings, without treating `a+b` as equivalent to `b+a`. Different seeds do not guarantee disjoint examples, semantic independence or held-out task generalization.

This overlap is documented, not silently removed: changing the corpus partition would change the historical experiment. The existing tests remain mechanism checks. A future generalization study should freeze an example/task-disjoint split and selection protocol before training and report its own results.

## Reproduce the selected suites

Choose the corresponding existing Python interpreter for each lane; first print its Python version and package metadata. From the respective lab directory:

```sh
# L1_pretraining_loop, using the PyTorch lane
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_pretraining -v

# L9_evaluation, using the NumPy lane
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest test_eval -v
```

Compare test IDs, source hashes and package/build observations as well as pass/fail. A different environment should emit its own new record, rather than copying this one or just changing its date. The source hashes identify what was executed; they are not authenticated wheel provenance, an installation lock or proof of cross-platform bitwise identity. Full historical training curves, other labs, fresh installation, a joint package lane, GPU/MPS and distributed training were not run.

To repeat the exact-example overlap check from L1:

```python
from train import make_text
train = set(make_text(20000, seed=1).split(";")[:-1])
validation = set(make_text(2000, seed=2).split(";")[:-1])
print(len(train), len(validation), len(train & validation))
# 19777 1998 43 on the recorded Python build
```

## Selected-configuration worksheet

Before any larger run, fill this in alongside its results; do not populate unknown fields with a guessed default. A design-only capstone can mark its target execution as proposed/not-run and report small CPU checks separately; this worksheet does not require running a frontier-scale training job.

| Field | Required report |
|---|---|
| Status and source | Proposed versus executed; full source revision, dirty diff and input-code hashes |
| Environment | Exact Python, PyTorch, NumPy including absent packages; installed build/artifact identity, OS/CPU/GPU, driver/runtime and relevant thread/determinism settings |
| Data and roles | Dataset revision and content hashes; train/validation/test IDs or generator settings; overlap checks and what kinds of independence were tested |
| Seeds and sampling | All RNG families, seeds, sampler state and fixed sample/draw order; identify unseeded or unsupported components |
| Candidate budget | All predeclared configurations, seeds, steps/tokens/rollouts and maximum search/compute budget; include failed/rejected trials |
| Selection | Metric and split used for selection, checkpoint rule, tie rule and when selection stops; state if nothing was selected |
| Evaluation | Freeze the selected configuration before held-out evaluation; disclose contamination, exclusions, missing outcomes and number of evaluated examples |
| Evidence | Commands/configuration, logs, curves, checkpoint/output hashes and completed-work counters; distinguish correctness checks from quality and timing |
| Limits | Unsupported/not-run lanes; fresh-install and cross-platform status; remaining verification needed |

**Worked status for this update:** two existing CPU unit suites, not a hyperparameter search. L1 executes both fixed configurations and selects neither; its final validation assertion does not become an unbiased held-out quality result. L9 checks mathematical/synthetic fixtures, without selecting a trained model. No optimizer ranking follows from either lane.

## Primary documentation and remaining gate

[PyTorch 2.9 reproducibility notes](https://docs.pytorch.org/docs/2.9/notes/randomness.html) explain why the same seed alone does not guarantee results across releases, platforms or devices. [NumPy 2.3 RNG compatibility policy](https://numpy.org/doc/2.3/reference/random/compatibility.html) conditions stream compatibility on the generator, call sequence, build and environment. These documents support reporting those factors; they do not qualify our runtime or promise bitwise replay.

The bounded TRN-4 manifest/reporting work is complete here. A fresh isolated installation with a tested lock, a joint package environment, full historical-table reproduction and distributed/GPU qualification remain separate compatibility work. Existing PyTorch 2.15 / NumPy 2.5 reference claims stay explicitly historical; no new “latest” pin is inferred from the installed versions.
