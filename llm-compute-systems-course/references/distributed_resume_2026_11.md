# Two-rank checkpoint recovery pilot — 2026-11-14 UTC

CMP-5 now has a pinned TorchTitan/FSDP2 source path and an actual two-process CPU DDP/Gloo/DCP restart fixture. Three seeds pass exact uninterrupted-versus-resumed comparisons. This is not execution of TorchTitan or FSDP2, a GPU recovery result, an elastic/topology-change test, crash durability or a process-kill experiment. CMP-5's target qualification remains open.

## Frozen execution contract

[Protocol](../labs/kit/distributed_resume_protocol.json) committed at `df3ce35`, implementation at `1e0be52`, control correction at `d9ad23c`. Seeds11/23/37, world size2, Gloo, CPUfloat64, one thread per worker. The tiny Linear4→8/ReLU/Dropout.3/Linear8→2 model trains six updates with AdamW lr.01 and eight examples/rank/update. No model quality comparison, search or seed selection occurs.

A baseline job performs all six steps. A second job stops cleanly after step3 and saves real DCP model/optimizer state plus rank-local cursor, Python/NumPy/Torch RNG state. Both workers exit. A third fresh launcher spawns two new workers, verifies checkpoint identity/completion and loads state before steps4–6. Final model hashes agree between ranks. There are 96 unique example IDs, partitioned across rank and cursor; independent accounting verifies exactly IDs0–95 once in the clean resumed path. Random perturbations use Python and NumPy RNG; dropout uses Torch RNG. Data/input hashes and full per-rank loss sequences match the baseline exactly, as do final model, optimizer, RNG, cursor and completed-step hashes/values.

The checkpoint uses synchronous `dcp.save/load` with `get_state_dict/set_state_dict` and an App Stateful wrapper. Rank-local JSON sidecars preserve RNG/cursor state. After both saves and sidecars complete, rank0 publishes a hash manifest, then both ranks synchronize. The loader rejects missing completion/sidecar, altered DCP bytes, wrong topology, protocol or runner. The local pending-file rename is a visibility convention; no fsync, object-store atomicity, adversarial authentication, partial-save crash or power-loss claim is made. Only course-created local checkpoint bytes are loaded. DCP's format is not promised compatible across Torch versions.

## Results and useful-work boundaries

| Seed | Restored rank comparisons | Max rank save/barrier stall s | Max rank load/barrier s | Prefix + fresh-resume process s | Unique examples | Checkpoint bytes |
|---|---|---|---|---|---|---|
| 11 | 2/2 exact | 0.053076 | 0.062826 | 2.878752 | 96/96 | 198611 |
| 23 | 2/2 exact | 0.060342 | 0.060537 | 2.862765 | 96/96 | 198613 |
| 37 | 2/2 exact | 0.051235 | 0.061602 | 2.802467 | 96/96 | 198506 |

[Raw summary](../labs/kit/examples/distributed_resume/summary.json), [work accounting](../labs/kit/examples/distributed_resume/work_accounting.json), and [all raw ranks/logs/checkpoints](../labs/kit/examples/distributed_resume/) retain the evidence. Each rank has the full step trace (IDs, loss, input digest and step interval), final state digests, save/load intervals and total worker time. All 12 corrected fresh-launcher jobs took 16.936593s combined, including three omission runs on seed11. Each launcher spawns two actual workers; this is 24 worker processes, not 12 ranks or a full model benchmark.

Save stall includes local DCP write, sidecars, completion publication and barriers. Load time includes digest verification, DCP/sidecar restore and barrier. The larger parent intervals include process startup/import/teardown. Rank intervals overlap: do not sum them to invent job wall time. The clean restart has zero repeated IDs; it does not estimate work lost during a real fault. Startup dominates this tiny CPU model, so no throughput speedup, optimal checkpoint interval or production efficiency is inferred. Raw step times remain descriptive and do not subtract save overhead from an alleged GPU timeline.

## Controls, including the initially ineffective omission

On seed11, omitting RNG produces different inputs/losses/final state; omitting optimizer destinations produces different post-restart losses/model/moments; rewinding the cursor repeats/skips IDs and changes the trajectory. Both ranks detect every omission. The six artifact guards reject invalid inputs before DCP load. [Retained verification](../labs/kit/examples/distributed_resume/validation.json) checks checkpoint/source identities and all six rank comparisons. Three existing MoE/recovery tests also pass, preserving the original single-process fixture.

The first implementation's optimizer omission was ineffective: `dcp.load` had already populated live optimizer tensors returned by `get_state_dict`, even when its setter was skipped. The suite correctly stopped because the negative control did not diverge. [Initial evidence/logs](../labs/kit/examples/distributed_resume/first_attempt.json) retains that result. The corrected control excludes optimizer destinations from the load state dictionary altogether; the entire three-seed suite was rerun in a new directory with no overwritten evidence or protocol tuning. This is also a concrete demonstration of DCP's documented in-place loading contract.

Runtime: CPython3.12.14 / Torch2.9.1 / NumPy2.3.5, macOS27.0.1 arm64. Used the separately tested training-course joint CPU environment; no compute default requirements/tracker migration. Logs retain the macOS redirect-support note and DCP TypedStorage deprecation warnings. Real Gloo local communication ran; no GPU/NCCL/DTensor/FSDP process was invoked.

## Pinned author path: read, not executed

TorchTitan commit [`631d544b1f6794ecfeca734a7bba861e03ddd5e3`](https://github.com/pytorch/torchtitan/tree/631d544b1f6794ecfeca734a7bba861e03ddd5e3). [Selected file hashes](torchtitan_resume_sources_2026_11.json) identify the inspected sources. This is a new dated reading, separate from the older training-stack walkthrough; no package installation or author test execution is claimed.

- `torchtitan_recipes/tests/models/llama3.py::llama3_debugmodel` constructs debug Llama3, AdamW lr8e−4, a two-step warmup/linear scheduler, ten training steps, Grain C4-test packing and no validator. Crucially, the default recipe has `checkpointer=None`: it is not a ready-made restart qualification run.
- `training_engine.py::_initialize_model` calls model parallelization. Llama3 inherits Decoder; `models/common/decoder.py::_apply_fsdp` resolves mesh and dispatches to `distributed/fsdp.py::apply_fsdp_to_decoder`, which wraps decoder blocks/root with `fully_shard`. `config/parallelism.py` distinguishes DP replication from sharding. A proposed target gate fixes DP-shard2/DP-replicate1, TP/PP/CP1 and a full checkpoint at step3; that proposed configuration has not been instantiated here.
- `trainer.py` passes the dataloader in extra states, loads a checkpoint before building its microbatch iterator, and saves after completed updates. `training_engine.py::_initialize_checkpointer` registers optimizers, schedulers and train state. Its `state_dict` stores step and tokens seen; it does not itself serialize all RNG streams. This selected reading cannot establish global RNG completeness; the actual target run must identify the owner for every random stream rather than infer recovery from a step field.
- `components/data/loader.py::GrainDataLoader.state_dict/load_state_dict` records version, effective DP degree and rank-local iterator state, rejects changed DP degree or missing rank state, and restores the iterator. DCP tensor resharding support alone therefore does not establish application-level dataloader portability across topology changes.
- `components/checkpointer/dcp.py::dcp_save` separates sync from async/process-staged writes. Pinned-mode staging and upload futures have distinct waits; `training_engine.py::optim_step` waits for staging before optimizer mutation. Close/wait paths matter: an async submission or staged buffer is not proof the final checkpoint is complete.
- The author checkpoint README separates full resumable step directories from model-only final exports, and documents excluded loading keys. Weight-only export is not optimizer/data/RNG recovery. We do not claim this reading exhausts TorchTitan's full optional-state ecosystem.

Primary docs inspected 2026-11-14: [PyTorch2.9 DCP API](https://docs.pytorch.org/docs/2.9/distributed.checkpoint.html) (parallel and in-place load, version compatibility warning), [current FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) (rendered2.15, distinct from our runtime). The source hash ledger plus pinned repository links avoid treating moving tutorial defaults as the tested environment.

## Reproduce and remaining acceptance

From `labs/kit`, under the recorded CPU environment:

```bash
python distributed_resume.py suite --directory /tmp/new-cpu-resume-evidence
```

The output directory must be new. Local two-process communication is required. Evidence includes binary DCP artifacts and rank sidecars; preserve them with the source/protocol hashes. No actual crash/fault rate was sampled. Next qualify the pinned author/config/dependency stack on supported accelerators: uninterrupted/resumed full progress (including schedule, accumulation, worker queues and every RNG stream), interrupted-save exclusion, real failed-worker restart, shard/layout behavior where application data supports it, save stalls and useful completed tokens. Until then, CMP-5 FSDP2 recovery remains unqualified; existing CPU sharding arithmetic and this DDP pilot cannot substitute for it.
