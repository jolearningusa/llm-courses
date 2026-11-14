# ExecuTorch old/new CPU artifact matrix — 2026-11-14 UTC

CMP-1's bounded isolated **unquantized XNNPACK CPU compatibility pilot** is now executed. Fresh ExecuTorch **1.0.0 / Torch 2.9.1** and **1.6.0 / Torch 2.15.0** environments both exported/lowered the same frozen MLP. Each trusted locally built `.pte` was then loaded in **both** consumer environments, in a new process per matrix cell. All four cells passed three eligible input cases, the scalar numerical oracle and before-native input guards.

This qualifies the named model/contracts on **CPython 3.12.14 / macOS 27.0.1 arm64 / CPU**, not arbitrary `.pte` programs, a general ABI promise, LLM throughput, quantization, Core ML, CUDA, edge hardware or native C++ application integration. The historical default requirements/tracker still identify the original 1.0.0 course lane; the tested 1.6.0 cohort has a [separate installed snapshot](../labs/kit/requirements-executorch-1.6.0-cpu-lock.txt). No automatic migration or historical-table recertification is implied.

## Frozen inputs and independent oracle

The [protocol](../labs/kit/executorch_compat_protocol.json) and [literal float32 fixture](../labs/kit/executorch_compat_fixture.json) were committed at `8c0e5c0` before export/runtime testing. The [runner and two controls](../labs/kit/executorch_compat.py) were committed at `9c88367` before builds. Weights and all three `(3,8)` inputs are stored as float32-rounded JSON numbers, rather than relying on each Torch release producing the same RNG stream. Both models are Linear(8,16) → ReLU → Linear(16,4), eval mode, CPU float32, one Torch thread, default XNNPACK partitioner options and static shapes.

The eligible cases are `random`, `zero` and `signed`. A separate standard-library scalar double-precision MLP computes expected results from the same frozen float32 values; eager, exported and native outputs must satisfy **rtol 1e-5 / atol 1e-6**. A hand-computed two-input example checks the scalar oracle's ReLU/bias behavior. Tensor-byte input identities are compared across both producers and all consumers. No training, model-quality selection or best timing was performed.

The existing L11 [`deployment_lab.py`](../labs/kit/deployment_lab.py) is unchanged; its `checked_forward` enforces CPU/float32/contiguous/static `(3,8)` before executing a callable. A spy independently checks that wrong shape `(4,8)`, float64 dtype and non-contiguous `(3,8)` stride all raise **without calling native execution**. Both producers also exercise the exported module's own static-shape rejection after valid exported cases pass. The exact release-specific exception is retained in the report rather than assuming a stable exception type.

## Artifact and partition evidence

| Producer | Artifact bytes | SHA-256 | Actual delegated segments | Remaining ATen compute fallback |
|---|---:|---|---:|---:|
| 1.0.0 | 3,728 | `a910c25afff67ecab8c68a9c3663d041d7f720d084f694d1831e36660aa3337d` | 1 `XnnpackBackend` | 0 |
| 1.6.0 | 3,728 | `424a6e0b1e4a02c0c7bbc8b13b924d3d15c13cd920bd774fe215a9ca862ff764` | 1 `XnnpackBackend` | 0 |

The two artifacts share a suffix and byte count but have **different hashes**. Compatibility is established by the actual load/execute cells, not by suffix, size or version numbers. The [full record](executorch_cpu_compatibility_2026_11.json) retains edge graphs, call targets, delegate IDs, fallback lists, artifact hashes and all outputs. `getitem` and delegate plumbing are not counted as ATen compute fallback. These records do not authenticate arbitrary candidate-generated logs or prove all native operators are safe.

## Fresh-process producer / consumer results

“Producer” identifies the exporter/lowerer; “consumer” identifies the imported runtime. Every cell starts a new Python process, loads the existing artifact and reconstructs only input tensors/scalar expected values, **without constructing or exporting a model**.

| Producer | Consumer | Eligible inputs passed | Worst absolute error vs scalar | Program + method load ms | Native calls for invalid inputs |
|---|---|---:|---:|---:|---:|
| 1.0.0 | 1.0.0 | 3 / 3 | 5.96046448e-08 | 0.426 | 0 |
| 1.0.0 | 1.6.0 | 3 / 3 | 5.96046448e-08 | 1.664 | 0 |
| 1.6.0 | 1.0.0 | 3 / 3 | 5.96046448e-08 | 0.273 | 0 |
| 1.6.0 | 1.6.0 | 3 / 3 | 5.96046448e-08 | 0.311 | 0 |

The largest observed error is **5.960464477539063e-08**. All three fixture outputs also agree in both eager/exported producer lanes. The four cells report the exact producer artifact hash, and all input hashes agree with the build records. Reverse loading (1.6-produced artifact in 1.0) happened to pass this fixture; do not generalize it to arbitrary newer schemas/operators.

Each cell records the first call for each case, three warmups and ten warm host-call samples. Only the first case's first call is the process's first native execution; later case-first calls already use an initialized runtime. “Cold” here means a fresh process's import/load/first-use boundary, **not** OS-cache eviction, a clean machine or cold target boot. Export, lower/serialize, load/program-method, case-first, warm-call and full process phase intervals remain distinct. Tiny local timings, including the table's load observations, are not a stable old/new speed ranking, LLM benchmark or energy/memory measurement. No timing samples were dropped or selected by minimum.

## Fresh installations and primary-source identity

Both venvs exclude system site-packages. The old lane installed the existing [historical complete requirements](../labs/kit/requirements-executorch-lock.txt); the candidate installed `executorch==1.6.0` with its declared Torch **2.15.0** dependency, then saved all observed package versions in the separate candidate snapshot. `pip check` passes in both lanes. The snapshots include platform-specific/transitive dependencies; they are not hermetic cross-platform images. Pip built some pure-Python source distributions (recorded install logs/artifact URLs), rather than claiming every dependency came from a binary-only lock.

Actual pip installation reports record package URLs and archive SHA-256 values. The selected **Torch and ExecuTorch** artifacts' digests were independently compared with version-specific primary PyPI metadata. This does not constitute a full dependency source/security audit. The source references are:

- Baseline tag `v1.0.0` → [`8c84780911e7f4e9eb19395181b04761392a4b56`](https://github.com/pytorch/executorch/tree/8c84780911e7f4e9eb19395181b04761392a4b56).
- Candidate tag `v1.6.0` → [`4776f71b6f0daa5dfc7d758d5836fc1cc0d40154`](https://github.com/pytorch/executorch/tree/4776f71b6f0daa5dfc7d758d5836fc1cc0d40154).
- For **each** cohort, the actual installed `runtime/__init__.py` and `backends/xnnpack/partition/xnnpack_partitioner.py` bytes match those exact upstream source files. Their hashes are in the record. This is a selected Python-file source pair, not full-tree/native-build reproducibility. Bundled `flatc` binary hashes/version output were observed after the runs and are separately labeled in the record; the runner selects the environment’s bundled binary when none is on the inherited PATH.

[Versioned 1.0 getting-started](https://docs.pytorch.org/executorch/1.0/getting-started.html) documents the export/partitioner/runtime path. The web reader still could not read the versioned 1.6 page; instead we inspected the [getting-started text at the candidate commit](https://github.com/pytorch/executorch/blob/4776f71b6f0daa5dfc7d758d5836fc1cc0d40154/docs/source/getting-started.md), selected runtime API and partitioner source. No moving `stable` page is treated as proof of the installed 1.6 APIs.

## Warnings and scope limits

Raw build/consumer/install logs are retained in the JSON record. The 1.0 lane disables TorchAO C++ extensions for its Torch-version combination and reports **InternalConsistency verification requested but unavailable** during native loads. Our numerical/explicit input controls do not replace that missing verifier. The 1.6 consumer emits `Number of efficient cores 0`; this log is not a measured hardware topology statement. Absence of the old verifier warning does not establish the newer build's verifier coverage. Build logs also retain macOS multiprocessing-redirection and Enum-registration deprecation notes.

No quantized TorchAO or Core ML path ran, even though those packages are dependencies. No malformed binary, arbitrary native shape, sanitizer, fault-injection, target-driver or firmware test ran. Only trusted locally generated `.pte` artifacts were supplied to the native loader. No memory-footprint, race, crash-durability or cross-device guarantee follows from these checks.

## Replay

From the repository root, use separate fresh CPython 3.12 environments and new output paths. The tested host build is recorded in each report; different platforms need their own package cohort.

```sh
python3.12 -m venv /tmp/et-old
python3.12 -m venv /tmp/et-new
/tmp/et-old/bin/python -m pip install -r llm-compute-systems-course/labs/kit/requirements-executorch-lock.txt
/tmp/et-new/bin/python -m pip install -r llm-compute-systems-course/labs/kit/requirements-executorch-1.6.0-cpu-lock.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/et-old/bin/python -I \
  llm-compute-systems-course/labs/kit/executorch_compat.py build \
  --directory /tmp/et-old-artifact --report /tmp/et-old-build.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/et-new/bin/python -I \
  llm-compute-systems-course/labs/kit/executorch_compat.py build \
  --directory /tmp/et-new-artifact --report /tmp/et-new-build.json
# Repeat with both Python interpreters for each producer directory; use a new report each time.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/et-new/bin/python -I \
  llm-compute-systems-course/labs/kit/executorch_compat.py load \
  --directory /tmp/et-old-artifact --report /tmp/et-old-in-new.json
/tmp/et-old/bin/python -I -m unittest discover \
  -s llm-compute-systems-course/labs/kit/tests -p test_executorch_compat.py -v
/tmp/et-new/bin/python -I -m unittest discover \
  -s llm-compute-systems-course/labs/kit/tests -p test_executorch_compat.py -v
```

The runner refuses an existing build directory/report rather than overwriting evidence, records failed Python stages separately and loads trusted local producer artifacts. A subprocess signal/timeout would require preserving its raw exit/log as a failed or incomplete cell; no such event occurred here. Runtime wrapper guards are not security isolation. Replays are additional reproducibility observations, not independent model/task coverage.

**Completed:** CMP-1's bounded two-environment CPU export/lower/partition/load/execute/rejection/cold-warm evidence, including explicit cross-version cells. Default-pin migration and broader models/backends remain separate decisions. Other bounded audit items remain open. AGT-2's later PR #151 supplies a measured strict-score pilot, while broader semantic/human/large-cohort work remains pending; PR #120 itself still covers teaching/protocol only.
