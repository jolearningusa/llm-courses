# Fresh joint CPU installation — 2026-11-14 UTC

TRN-4's fresh-install/tested-lock gate now has one executed **CPython 3.12.14 / macOS 27.0.1 arm64 / CPU** cohort. It joins PyTorch **2.9.1** and NumPy **2.3.5** in a newly created venv without system site-packages. All **99 existing tests** across L0–L9 passed, with no skips, errors or failures. This is unit-suite compatibility, not reproduction of the historical PyTorch 2.15 / NumPy 2.5 result tables, new held-out quality, performance, GPU/MPS or distributed qualification. The [earlier two-lane report](cpu_environment_evidence_2026_11.md) remains a dated observation.

## Artifact → install → execution

1. CPython 3.12.14 created a new venv in a previously unused directory. Bootstrap pip was **25.0.1**; the host-provided Python/pip/OS are recorded, not recreated by the runtime-package lock.
2. `pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: torch==2.9.1 numpy==2.3.5` resolved **11 runtime distributions**. All selected wheel filenames, byte counts, SHA-256 values and primary PyPI file URLs are in the [artifact/install record](cpu_lock_artifacts_2026_11.json). Both direct and transitive dependencies are explicit in the [platform lock](../labs/requirements-cpu-macos-arm64-py312.lock). No source build ran. The lock was committed before suite execution.
3. Installation used the local wheel directory, `--no-index --only-binary=:all: --require-hashes -r …`; the actual pip installation-report hashes equal the downloaded hashes, independently compared with PyPI metadata for every wheel. [Download log](cpu_lock_logs_2026_11/download.txt), [installation log](cpu_lock_logs_2026_11/install.txt), [pip check](cpu_lock_logs_2026_11/pip-check.txt). Hashes identify a chosen artifact; they do not audit upstream code or lock the host ABI/OS.
4. [The runner](../labs/cpu_qualification.py) checks platform/versions, then uses **one fresh Python process per lab**, with that lab's own import directory. This prevents modules named `model` or `train` from leaking between labs. Existing tests, data generation, tolerances and training budgets were unchanged. Each worker records its test IDs, counts, source hashes, result categories and raw output in the [execution manifest](cpu_lock_manifest_2026_11.json).
5. A literal float64 `torch.from_numpy()` shared-storage mutation and `.numpy()` reverse mutation both passed; imports alone were not the bridge check. `pip check` found no broken requirements.

The tested runner/source commit is `a4430b5be85b96b8ad23dcb66e33f49a40d27a4a`; its full hash and each lab `.py` digest are recorded. Later documentation/control-script changes do not change those lab sources. The runner does not authenticate arbitrary pre-existing package contents: its version guard complements, rather than replaces, the recorded fresh hashed installation.

## Existing suites actually executed

| Lab | Tests | Outcome | Raw worker output |
|---|---:|---|---|
| L0 | 10 | passed | [L0_training_budget.txt](cpu_lock_logs_2026_11/L0_training_budget.txt) |
| L1 | 10 | passed | [L1_pretraining_loop.txt](cpu_lock_logs_2026_11/L1_pretraining_loop.txt) |
| L2 | 10 | passed | [L2_data_pipeline.txt](cpu_lock_logs_2026_11/L2_data_pipeline.txt) |
| L3 | 7 | passed | [L3_sft_packing.txt](cpu_lock_logs_2026_11/L3_sft_packing.txt) |
| L4 | 9 | passed | [L4_lora_nf4.txt](cpu_lock_logs_2026_11/L4_lora_nf4.txt) |
| L5 | 7 | passed | [L5_reward_ppo.txt](cpu_lock_logs_2026_11/L5_reward_ppo.txt) |
| L6 | 7 | passed | [L6_dpo.txt](cpu_lock_logs_2026_11/L6_dpo.txt) |
| L7 | 20 | passed | [L7_grpo.txt](cpu_lock_logs_2026_11/L7_grpo.txt) |
| L8 | 11 | passed | [L8_distillation.txt](cpu_lock_logs_2026_11/L8_distillation.txt) |
| L9 | 8 | passed | [L9_evaluation.txt](cpu_lock_logs_2026_11/L9_evaluation.txt) |
| Total | **99** | **passed** | Per-test IDs in manifest |

The L3 test emits PyTorch's warning about converting a `requires_grad=True` tensor to a scalar (`test_sft.py:50`). It is preserved in the log; the assertion passes. The earlier missing-NumPy warning is absent from this joint cohort. No warning was suppressed.

Workers use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTHONHASHSEED=0`, and observed Torch thread count 1. NumPy reports Accelerate BLAS with version unknown; those environment variables do not prove Accelerate's thread count. Deterministic algorithms remain disabled as in the original tests, and the wrapper does not add global Python/NumPy/Torch seeds. Existing per-test seeds remain unchanged; unseeded operations and elapsed suite times are not bitwise-replay or performance claims. Each suite's timing excludes worker startup/import overhead.

## Acceptance controls

[Five runner controls](cpu_lock_runner_controls_2026_11.json), reproducible with [the control script](../labs/check_cpu_qualification_controls.py), accepted a legal literal test and rejected an empty suite, assertion failure, skipped test and mocked Torch-version drift. Drift was rejected **before creating an evidence directory**. These are independent tiny fixtures, not failures in the 99 lab tests. Existing output directories are refused to preserve prior evidence.

[Two installer controls](cpu_lock_installer_controls_2026_11.json) used actual `pip install --dry-run --ignore-installed --no-index --require-hashes` against the local wheel directory: a zeroed NumPy SHA-256 was rejected, and omitting the transitive `mpmath` pin/hash was rejected. Nothing was installed by those negative controls. The main installation succeeded before them; no failed trial was removed from the test ledger.

## Reproduce this cohort

Use **CPython 3.12.14 on macOS arm64**. From the repository root, choose new paths for the venv, wheel directory and results; do not reuse an environment with unrelated packages. This lock contains only the selected macOS wheels and is deliberately not a Linux/Windows/x86 lock.

```sh
python3.12 -m venv /tmp/training-cpu-env
/tmp/training-cpu-env/bin/python -m pip --version
# Recorded bootstrap: pip 25.0.1. Record any different bootstrap in your result.
/tmp/training-cpu-env/bin/python -m pip --isolated download \
  --index-url https://pypi.org/simple --only-binary=:all: --require-hashes \
  -r llm-training-course/labs/requirements-cpu-macos-arm64-py312.lock \
  --dest /tmp/training-cpu-wheels
/tmp/training-cpu-env/bin/python -m pip --isolated install \
  --no-index --find-links /tmp/training-cpu-wheels --only-binary=:all: \
  --require-hashes --report /tmp/training-cpu-install.json \
  -r llm-training-course/labs/requirements-cpu-macos-arm64-py312.lock
/tmp/training-cpu-env/bin/python -I llm-training-course/labs/cpu_qualification.py \
  --output /tmp/training-cpu-results
/tmp/training-cpu-env/bin/python -I llm-training-course/labs/check_cpu_qualification_controls.py \
  /tmp/training-cpu-controls.json
```

Unset `PYTHONPATH`/`PYTHONHOME`; the runner rejects them. `-I` prevents user-site and environment-based Python import additions. Bootstrap is outside the runtime lock and wheel availability can change: archive the wheel directory separately if durable offline recreation matters. The committed artifacts are identities and logs, not bundled binaries or a fully hermetic system image. A second independent fresh install was not run; this is one executed installation plus its acceptance controls.

## Reading and completion boundary

Primary sources re-read on 2026-11-14 UTC: [PyTorch previous versions, 2.9.1 / OSX](https://pytorch.org/get-started/previous-versions/#v291), [PyTorch 2.9 reproducibility notes](https://docs.pytorch.org/docs/2.9/notes/randomness.html), [NumPy 2.3 RNG compatibility](https://numpy.org/doc/2.3/reference/random/compatibility.html), and [pip hash-checking installs](https://pip.pypa.io/en/stable/topics/secure-installs/). The first supports the selected older OSX package cohort; the next two limit seed/replay claims; pip documents exact pins and hashes for every dependency. Our actual installation/tests establish the bounded course observation, not those documents alone.

This completes the bounded **TRN-4 fresh joint CPU installation / tested runtime lock**. Prior manifest/selection teaching is retained. Historical full-run tables, cross-platform qualification, newer-package migration, GPU/distributed checks and AGT-2's measured matched-memory benchmark remain open; none is silently deferred or declared complete. L1's documented 43 shared exact train/validation examples remain unchanged, so passing its loss-fall test does not become disjoint held-out task quality.
