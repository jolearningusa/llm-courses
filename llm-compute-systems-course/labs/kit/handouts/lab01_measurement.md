# L1 — Measurement and host data paths

**Released path:** CPU timing and analytical pipeline/interval exercises. Python 3.9+, standard library only. A CUDA trace extension is specified below but is not supplied or GPU-validated by this kit. Expected time: 3–5 hours, pending a student pilot.

## A. Measure actual CPU work

From the kit directory, run:

```sh
python3 measurement.py --iterations 20000 --repeats 15 --warmup 3
python3 measurement.py --iterations 40000 --repeats 15 --warmup 3
python3 -m unittest discover -s tests -v
```

Save the JSON records. Explain what the wall timer includes, why the scalar loop is not a GEMM/GPU benchmark, and what the warmup excludes. Compare raw samples, median and nearest-rank p95. Do not assert a fixed scaling factor; record process/thermal interference and small-sample limits. Add a comparison that changes an implementation detail while preserving the checksum, then measure it with `benchmark`.

## B. Reason about overlapping intervals

The output includes deliberately constructed intervals in milliseconds: copy 0–4, kernel A 2–7, kernel B 8–10. Their union covers 9 ms of a 10 ms window even though their durations sum to 11 ms. Explain the uncovered interval. Construct a nested-overlap case and one clipped by the observation window; verify the helper result by drawing a timeline.

These intervals are analytical teaching fixtures, not observed device activity. Union coverage does not prove why an interval is uncovered or identify the critical path. Describe the stream, dependency and CPU events needed for causal attribution.

## C. Host pipeline budget

Assume CPU preparation takes 6 ms, transfer 2 ms and compute 4 ms per item. Derive serial latency, ideal steady-state period, and the effect of reducing compute to 2 ms. Then make CPU preparation variable (2, 2, 14 ms repeating). Explain how prefetch depth can change jitter without eliminating the long-run preparation bottleneck, and why queue storage matters.

## D. Optional real-device extension

On a supported CUDA host, obtain your own focused timeline and follow the official CUDA/Nsight guides in the bibliography. Measure unprofiled completed-operation wall time separately from device events, compilation/warmup and profiler runs. Inspect one suspected preparation, transfer, launch or allocation gap. Change one mechanism at a time and verify output and buffer lifetimes. Record exact hardware/software and commands; never substitute the constructed intervals above for a collected profile.

Without hardware, submit A–C and explicitly identify the missing evidence for D. A CPU analytical submission is a complete version of the released path, not proof of CUDA performance.

## Submit

Use the [report template](../../report_template.md). Include raw JSON, environment, work boundaries, output checks, interval diagrams, pipeline assumptions and one falsifiable starvation hypothesis. Compare the alleged gain with timing dispersion. Passing unit tests checks accounting and warmup logic; it does not guarantee a representative benchmark.
