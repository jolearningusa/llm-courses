# Inference bottleneck diagnosis worksheet

Optional extension to L1/L8/L10, paired with [deck 36](../../slides/36_inference_bottlenecks.html) and the [literature review](../../references/inference_bottlenecks_review.md). This is an experiment protocol, not a released GPU harness. A supported GPU/engine/profiler is required for GPU conclusions. Without one, submit predictions and a source/analytical report.

## 1. Fix the computation and observation boundary

Record model/tokenizer revisions, target and memory capacity, driver/compiler/runtime/engine versions, precision, context layout, backend dispatch, sampling parameters and quality constraints. Choose one sequence-latency experiment and one open-loop service-load experiment. State prompt/output lengths, arrivals, cache cold/warm identity, warmup, timing/drain rules and failed/unfinished-request accounting.

Record output projection, logits processing, top-k/top-p filtering, RNG/token choice and host delivery separately when the engine exposes them. A combined range must be labeled combined rather than assigned wholly to selection. Keep profiler evidence separate from final unprofiled timing.

## 2. Predict before profiling

| Hypothesis | Controlled change | Evidence that would support it | Evidence against it |
|---|---|---|---|
| Weight traffic limits small-batch linear layers | Increase actual decode batch with the same precision/context | Weight traffic amortized across rows; throughput improves before iteration cost grows strongly | Attention, host or communication already dominates |
| KV dominates long-context decode | Increase context with batch fixed | Attention share and KV traffic rise consistently with the contract | Another stage explains the added time |
| Host submission overhead is exposed | Supported eager versus graph replay | Host-correlated gaps shrink; full decode latency improves | GPU already has ready work; gaps are device/collective dependencies |
| Selection is significant | Eligible optimized sampler versus reference, same semantics | Selection time shrinks and the full path improves consistently | Output projection or other kernels dominate; no end-to-end change |
| Capacity constrains service batching | Increase offered concurrency, inspect memory/admission | Eviction, recompute, rejection or queues rise at capacity | Spare capacity and a separate execution bottleneck |

Changing offered concurrency does not guarantee a particular instantaneous engine batch. Save observed batch sizes and iteration events. Do not use tiny/large batch as a synonym for latency/throughput without measurement.

## 3. Minimal run matrix

1. Begin with a model that fits weights, workspace and KV. Use short/medium/long contexts and low/medium/high observed batches; determine sizes from a fit pilot rather than OOM-driven guesses.
2. Measure prefill and decode separately, then the full request. Time linear layers, attention, projection, selection, exposed communication and host intervals. Do not sum overlapping durations as wall time.
3. Change one eligible factor: supported precision, replay, sampler or scheduler configuration. Fix other settings and rerun at least three independent final measurements.
4. Repeat an adverse regime and one held-out workload. Validate outputs/quality and sampling semantics before timing credit. Graph replay does not waive shape/buffer/lifetime contracts.
5. Report TTFT, per-request TPOT plus token-gap distribution, aggregate output throughput, SLO goodput, memory peaks and all failures. Define one-output-token TPOT as undefined and report its population separately.

## 4. Selection-specific controls

For top-k: declare whether exact indices, sorted output and deterministic tie-breaking are required. For top-p: declare normalized input, threshold inclusion and combined k/p order. Compare current optimized implementations, not only a full-sort reference. Include peaked/flat distributions, duplicate logits, tails and masked tokens. Handle all-masked/nonfinite inputs according to the declared API contract.

For stochastic sampling, test allowed support and empirical distribution with a declared statistical test; identical seeds across different algorithms need not yield identical samples. For beam search, use tiny exhaustive reference cases with scores, ancestry, EOS, length penalties, stopping and KV ownership. Compare implementations of the same beam semantics; greedy versus beam is a workload/quality comparison, not a semantics-preserving speedup.

Sparse-attention top-k and MoE routing require separate fixtures and reference computations. A faster vocabulary sampler does not establish either path's correctness.

## 5. Report the decision

Provide a bottleneck hypothesis, one budget, raw samples/events, a confirming trace, measured versus predicted terms, one adverse case and the intervention decision. Use a measured baseline fraction for any Amdahl estimate; a faster component does not guarantee a service gain under queueing. Label every constructed estimate and unavailable target. The [experiment report](../report_template.md) supplies the common structure; [P26](../../projects/COURSE_APPENDIX.md#p26) expands this into a six-week project.

<a id="bottleneck-teaching-checkpoint"></a>
## CPU teaching checkpoint — deck 36

Allow 10–15 minutes (estimate). A constructed, nonoverlapping serial decode step takes 10 ms. Predict the benefit of a 5× faster token-selection stage when it initially takes (a) 1 ms or (b) 5 ms. Which baseline fraction matters more than implementation difficulty?

From `labs/kit/`:

```sh
python3 - <<'PY'
for selection_ms in (1, 5):
    baseline_ms, component_speedup = 10, 5
    candidate_ms = baseline_ms - selection_ms + selection_ms / component_speedup
    print('selection_ms', selection_ms, 'candidate_ms', candidate_ms,
          'speedup', round(baseline_ms / candidate_ms, 3))
PY
```

Expected: `selection_ms 1 candidate_ms 9.2 speedup 1.087` and `selection_ms 5 candidate_ms 6.0 speedup 1.667`. These are constructed Amdahl calculations, not GPU timings. A 5× sampler improvement therefore need not imply a 5× decode improvement. An overlapping stage needs critical-path analysis; loaded-service queueing needs its own measurements.

Revisit the deck's weight budget: 8 billion BF16 parameters occupy 16 decimal GB; at an assumed effective 2 decimal TB/s, one ideal full read takes 8 ms. Do not substitute this assumption for a measured bandwidth, or sum it blindly with overlapping profiler ranges. Compare [the inference analysis](https://jax-ml.github.io/scaling-book/inference/) and the existing annotated review.

Submit both predictions and outputs, then use sections 1–5 to choose one falsifiable hypothesis. Identify one observation that would **reject** selection as the next optimization target, such as projection dominating the combined range. Without a GPU, submit the proposed controls, missing evidence and explicit “not measured” cells. With a supported target, record baseline fractions and full-path results independently of this arithmetic.

中文提示：先估算原始占比，再讨论加速；采样难实现不代表它占主要时间。9.2 ms 和 6.0 ms 是构造值，不能当作 GPU 实测或排队服务收益。
