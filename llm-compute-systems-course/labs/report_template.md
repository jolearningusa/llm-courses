# Experiment report

## Question and prediction

State the training/inference/deployment objective, the suspected bottleneck and the trace signature predicted to change. Give one condition under which the intervention should lose.

## Environment and workload

Record device model/count/topology, memory, CPU/NUMA, driver/runtime/compiler versions, framework/library commits, selected kernels/backends, power/clock settings if relevant, checkpoint/model revision, tokenizer, tensor shapes, dtypes/scales, batch/context lengths and seeds. For serving, record arrival process, concurrency, cache state and timeouts. For training, record effective batch, optimizer, data order and checkpoint/recovery policy.

## Correctness and quality

Name the reference, tolerances, masks/stride/edge cases, gradient/loss or generation-quality checks. Explain any approximation or changed semantics. Record unsupported cases and fallback behavior. State how cache identity and tenant/session boundaries were checked where applicable.

## Measurements

| Variant | Build/cold start | Warm latency / useful throughput | Peak allocated/reserved memory | Communication/transfer | Quality/correctness |
|---|---|---|---|---|---|
| Baseline | | | | | |
| Intervention | | | | | |
| Ablation | | | | | |
| Adverse workload | | | | | |

Report repetitions and dispersion, measurement scope, synchronization, warmup and profiler overhead. Include raw result locations. Add TTFT/ITL tails, SLO attainment and failed requests for serving; time-to-target and recovery/lost-work accounting for training; cache token reuse, byte-seconds and recomputation for agent traces. Label simulated/analytical results separately.

## Explanation

Reconcile the prediction with measured data. Show the relevant timeline/kernel/collective evidence. Quantify how the component's time fraction bounds end-to-end benefit. Explain whether the bottleneck moved, whether a fallback changed the experiment and what further evidence would overturn the conclusion.

## Reproduction and limits

Provide exact setup/run instructions and artifact/configuration hashes, licenses/permissions for required data/weights, prerequisites and hardware skips. State what was measured locally, what came from a cited source and what remains a projection.
