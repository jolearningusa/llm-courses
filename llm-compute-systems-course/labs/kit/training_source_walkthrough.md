# Pinned training-stack source walkthrough

Reviewed 2026-10-26. These snapshots were inspected as source, not installed/executed as training stacks. Their dependencies are independent of the small torch 2.9.1 CPU lab.

## TorchTitan

Snapshot `e111f10d4db3e85e8f066922a7620ab685ca035d`: [repository](https://github.com/pytorch/torchtitan/tree/e111f10d4db3e85e8f066922a7620ab685ca035d), [training_engine.py](https://github.com/pytorch/torchtitan/blob/e111f10d4db3e85e8f066922a7620ab685ca035d/torchtitan/training_engine.py).

Read `TrainingEngine.Config.__post_init__`, `_initialize_distributed_runtime` and `_initialize_model`. The configuration validation computes an activation shard degree from sequence/context parallelism and checks token divisibility. Distributed initialization builds a parallelism context from configuration/topology. The model initialization path passes parallelization/compiler settings into model-specific transformations. Follow these calls rather than assuming a flag directly selects a kernel.

Student dossier: record the constraint expression, one caller/callee pair, logical mesh dimensions and the input/output layout around one model block. Find where the optimizer and checkpointer are initialized. Explain an ordering dependency. Avoid pasting pages of source; cite symbols and describe their role.

## Megatron-LM/Core

Snapshot `63f687dc83f44b31346e3775a5e23cf2337f7cbd`: [repository](https://github.com/NVIDIA/Megatron-LM/tree/63f687dc83f44b31346e3775a5e23cf2337f7cbd), [parallel_state.py](https://github.com/NVIDIA/Megatron-LM/blob/63f687dc83f44b31346e3775a5e23cf2337f7cbd/megatron/core/parallel_state.py).

Start at `initialize_model_parallel`, its tensor/pipeline/context/expert arguments and rank-group construction. Derive how data-parallel capacity follows the chosen model/context dimensions, then inspect the separate expert grouping constraints. A list of supported parallelisms is not permission to multiply all degrees independently.

Student dossier: draw two group memberships for one rank, identify a divisibility constraint and follow a schedule entry into one transformer/collective path. For a hardware run, add the exact environment and a real trace mapping; otherwise mark dispatch unobserved.

## Instructor prompts

A correct report links a configuration to specific symbols at the recorded commit, explains a rejected configuration and distinguishes source-derived behavior from runtime evidence. Common failures: reading moving main while citing a frozen commit, conflating activation checkpointing with restart checkpoints, or claiming a GPU backend was observed from a configuration flag alone.
