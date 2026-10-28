# L8 — Replay scheduling and budget disaggregation

**Released:** standard-library deterministic virtual-time simulation, plus analytical KV-transfer calculator. **Not run:** actual serving or split GPU cluster. Estimated 5–8 hours, student pilot pending.

From `labs/kit/` run `python3 scheduling_lab.py > scheduling-model.json` and the unit suite. Optional `--workload arrivals.json` accepts an array of `{id, arrival_s, prompt_tokens, output_tokens}` with unique IDs, finite nonnegative arrivals and positive integer token lengths. Costs in the output are constructed assumptions, not fitted hardware measurements.

## A. Chunking experiment

Predict which request will suffer an extended token gap when a long prompt arrives. Inspect the complete reference, then independently calculate the single-request timeline used in the test. Compare chunk budget 32 with an effectively unbounded budget on identical arrivals and one simulated colocated engine. At each iteration, requests already ready for decode emit one token, and pending prompts consume the prefill budget in arrival order. All outputs become visible at iteration end. Arrivals during a running iteration are noticed at the next boundary.

Check emitted-token conservation and nonnegative latencies. Report TTFT, per-request mean TPOT (undefined for one output token), completion time and iteration count. The model drains a finite workload, has no KV-capacity/cancellation/failure constraint, and reports zero failures by construction. It cannot establish production starvation freedom, throughput capacity or GPU overlap. Student task: add deadlines/rejections and account for every offered request rather than discarding slow requests.

## B. Transfer versus recompute

Use `transfer_seconds` to reproduce 512 MiB at assumed effective 50 GB/s plus 1 ms = 11.737 ms. Compare 10/50/100 GB/s and prefill-recompute assumptions 5/20/100 ms. Separate transfer, conversion, registration/queue overhead and missing-prefix compute. This is an analytical budget, not a full disaggregated-cluster simulation. Explain one case where recomputation wins and one where compatible transfer could win.

## C. Equal-resource experiment protocol

For an actual deployment, compare **two colocated workers** against **one prefill + one decode worker** with the same device count, total memory, topology and offered arrivals. Pin engines/connectors/model/tokenizer/dtype, precision and parallel layouts. Define SLOs before running, for example TTFT ≤200 ms and TPOT ≤30 ms; these are example targets, not measured capability. Include warmup separately, fixed observation window and drain policy. Record offered/completed/rejected/failed/timed-out/unfinished counts, goodput under both SLOs, queue time by stage, transfer bytes/time, recomputation and numerical checks. Inspect actual connector compatibility and failure handling.

Without hardware, submit a two-resource stage-capacity/queue ledger and the B sensitivity table with “not measured” deployment cells. Do not compare the one-engine replay directly against an imagined two-device split or claim a break-even calculation proves end-to-end speedup. Submit raw simulation JSON and [report](../../report_template.md).
