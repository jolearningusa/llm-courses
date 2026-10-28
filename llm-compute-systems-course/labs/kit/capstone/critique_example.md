# Critique an optimization claim

This is a **constructed analytical example**, not a student experiment or hardware result.

> Claim: The compute kernel is twice as fast, so the entire training job is twice as fast.

Given a disjoint 100 s ledger with 60 s compute and 40 s other work, halving compute produces 30 + 40 = 70 s. The justified model speedup is 100/70 ≈1.43×. This conclusion still assumes the remaining work is unchanged. Real overlap, changed batching, memory contention or numerical progression can invalidate that assumption.

A strong revision says: “Under this explicit disjoint-phase model, a 2× change to the 60% compute component predicts 1.43× useful-throughput improvement. We have not measured a device or validated the unchanged-remainder assumption.” It identifies the needed next experiment: matched useful tokens/data progression, correctness checks, raw wall times and a trace explaining the critical path.

Then ask: Were compile/startup, checkpoint/recovery and failed/repeated work counted? Does the changed kernel preserve the target computation and required gradients? Is the baseline a current eligible library? Does an adverse shape regress? Was the final test held out from tuning? A record labeled “analytical” cannot support a measured GPU or cross-vendor headline.
