# L13 — Speculative correction and agent-wait policies

**Released:** standard-library categorical sampling experiment and analytical single-wait accounting. **Unverified:** real draft/target decoding, engine cache control and multi-agent performance. Estimated 5–8 hours, student pilot pending.

From `labs/kit/` run `python3 agent_policy_lab.py > agent-policy.json` and the unit suite. The complete references are inspection baselines; derive the marginal and byte-second calculations independently before editing.

## A. Exact stochastic correction

Use p=[0.1,0.3,0.6], q=[0.5,0.4,0.1]. Accepted mass is min(p,q); rejected mass is 0.5. The normalized positive residual p−q puts all corrected mass on token 2, reconstructing p exactly. Check equal/disjoint support and zero-probability entries. The tests enumerate the marginal (stronger than sampling alone) and compare 20,000 deterministic-seed samples within 0.015 absolute frequency tolerance.

Explain why drawing a fresh token directly from p after every rejection is wrong. Derive expected progress 1+α+…+α^k only under the stated constant independent-acceptance model. Compare low/high acceptance and draft cost using explicit assumptions. This reference is **one categorical step**, not an autoregressive model decoder or a cache rollback implementation.

For a real extension, pin draft/target/tokenizer/sampling transformations, validate accepted-prefix/first-rejection state and distributional behavior, and compare off/fixed/adaptive speculation at matched load/resources. Include draft/verify/memory overhead, startup, failures and adverse high-load or low-acceptance cases. An acceptance percentage alone is not a speedup.

## B. Tool-wait state policy

Reproduce short/long wait cases for 512 MiB state, 30 ms recompute, 10 ms outbound and inbound copy, and 100 ms TTL. These costs are constructed assumptions. `wait_policy` reports idle HBM byte-seconds **only inside the tool-wait window**, extra resume seconds and transfer bytes. It excludes inbound/recompute residency, capacity, queueing, shared-prefix overlap, transfer contention and device execution.

Outbound offload may overlap the wait; if the tool returns before it completes, the model counts the remainder plus inbound restore. An identity mismatch detected at resume forces recomputation and prevents incompatible restore. A real engine may cancel/reclaim differently; this analytical model does not implement its protocol. Student task: add a finite-capacity multi-session event model with correct shared ownership, keeping its simulated results labeled.

Vary wait/recompute/transfer/TTL and identity. Submit a sensitivity table with retain, offload and recompute adverse cases. Link each policy decision to the [cache compatibility notes](../../../notes/agentic_caching.md). For real engine work, inspect supported lifecycle APIs and the [open RFC](https://github.com/vllm-project/vllm/issues/52113) separately; do not send invented hint fields or physical block pointers. Record task completion, fairness, avoided/recomputed tokens, byte-seconds and failures. Submit JSON and [report](../../report_template.md).
