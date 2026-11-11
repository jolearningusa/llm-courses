# Rollout → reward → learner: probability contracts — 2026-11-10

TRN-2's bounded source/CPU-contract portion, based on main `b82f894` after #137. Deck 20 slides 9/10 add original role and evidence diagrams. [L7](../labs/L7_grpo/) contains the executable exercise and [actual report](../labs/L7_grpo/rollout_consistency_report.json). Historical PPO/GRPO derivations and earlier quality tables are unchanged.

## Fixed author path; source inspection only

verl revision **`2b4ed6826a39205838d0ca004bdd942334dbe4b6`**; [file URLs and SHA-256 manifest](verl_reading_manifest_2026_11.json). These four files were downloaded as text, not installed/imported. This is a selected path through the standard trainer's `fit`, with GRPO-style advantages, no critic and `bypass_mode=False` as the **reading branch**, not a complete qualified job configuration.

| Source / inspected functions | Observed contract |
|---|---|
| [ray_trainer.py](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/verl/trainer/ppo/ray_trainer.py), `fit`, `_compute_old_log_prob`, `_update_actor` | Generation returns before this outer-loop path extracts rewards and establishes response masks. Decoupled mode recomputes `old_log_probs` once per batch as a fixed proximal anchor, then forms advantages and updates the actor. The helper handles padding conversions and converts recorded log-probs to float. Weight publication follows the actor update. Named generation/reward/log-prob/update/weight-sync timers are source instrumentation, not measured intervals here. |
| [rollout_corr_helper.py](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/verl/trainer/ppo/rollout_corr_helper.py), correction and bypass helpers | Decoupled correction uses `old_log_prob - rollout_log_prob`, distinct from the learner PPO ratio. Token- and sequence-level aggregation differ; configured truncation/rejection changes weights/masks. Bypass mode substitutes rollout log-probs as `old_log_probs` and selects its corresponding loss mode. Empty masks and incompatible shapes are checked. |
| [core_algos.py](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/verl/trainer/ppo/core_algos.py), `compute_policy_loss_vanilla` | The inspected loss exponentiates learner minus old-anchor log-probs, with a log-ratio stability clamp, clipping and response-mask aggregation. Other registered losses have other contracts; this reading does not execute or qualify the chosen engine/loss configuration. |
| [rollout_corr.md](https://github.com/verl-project/verl/blob/2b4ed6826a39205838d0ca004bdd942334dbe4b6/docs/algo/rollout_corr.md), overview/configuration | Identifies backend/precision differences and temporal weight lag as separate mismatch sources. Shared weights alone do not guarantee equal numerical distributions. Correction mode, reduction and sampling conventions must accompany any comparison. |

The async rollout-manager name does not by itself establish pipeline concurrency, queue capacity or staleness in this selected outer loop. Full asynchronous/partial-rollout paths and worker/backend internals were not audited. No Ray, inference engine or GPU was run.

## Name all three probability roles

Let `q` be the distribution that generated the stored response, `p_old` the fixed proximal anchor for the batch, and `p_theta` the evolving learner. On aligned valid response tokens, the PPO ratio is `exp(lp_theta-lp_old)`; decoupled rollout correction uses `exp(lp_old-lp_q)` before its configured transformations. Bypass mode changes this decomposition; do not silently swap denominators. A reference policy used for a KL term is another role.

The identity control requires the same weights **and** tokens, prompt/previous-token alignment, EOS convention, mask, temperature/filter convention and numerical path. State hashes aid correlation, not authentication or proof of distribution equivalence. Natural-log differences produce probability ratios. Sequence ratios exponentiate the **sum** of valid token log-ratios; exponentiating their mean is a different geometric normalization.

## Actual CPU control and limits

Python 3.12.14 / torch 2.9.1, CPU, float64, one torch thread; `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. NumPy is absent in this existing lane and torch emits an initialization warning. No new dependency lock or framework migration is claimed.

The unchanged L7 `Policy(hidden=16)` actually samples four responses with model seed 23 / sample seed 29, temperature 1 and no top-k/top-p transform. A temporary observer captures the logits used inside the original sampler. The report gathers sampled-token log-probabilities from those logits, then compares the independently batched `Policy.logprobs` and a stepwise replay. EOS itself is included; later padding is excluded using an independent literal first-EOS reference. The sampler forces EOS after completion; recorded model log-probs at those padding positions are bookkeeping only, not the forced padding distribution. There are 32 sampled slots and 25 valid response tokens.

The maximum rollout/anchor ratio error is `4.440892098500626e-16`; a cloned current/anchor comparison is exactly one. The four actual strict rewards are all zero. To isolate probability drift rather than pretend to learn from those rewards, one SGD update uses explicitly **synthetic signed advantages** `[-1,-0.5,0.5,1]`. It changes the learner hash and moves valid PPO ratios to about `[0.98085,1.00586]`; rollout and anchor hashes stay fixed. This is one controlled update, not a quality improvement, GRPO replication or policy-lag research result.

Eight new tests plus eleven existing non-long-training controls pass (19 total). They cover identity, stepwise/batched agreement, first-EOS masking, invalid/empty masks, temperature and previous-token mismatch, excluded padding, ratio units and weight changes. The metric helper selects valid positions before exponentiation, including its deliberate NaN-padding case; this does **not** certify the unchanged full policy loss against arbitrary nonfinite padding.

Single CPU windows for sampling, rule reward, anchor rescoring and update are retained separately. They include this teaching instrumentation and exclude cloning, fingerprinting and other setup; they are not end-to-end rollout throughput or GPU kernel measurements. One synchronous batch is in flight with zero waiting batches; these local counts are not verl queue measurements. Repeat reports match all deterministic evidence after excluding the four CPU timing intervals.

## Next execution gate

For an actual verl pilot, freeze the full config, checkpoint/tokenizer, backend and precision; log sample IDs, behavior/anchor/learner weight versions, temperature/filters, response masks and reduction. Record actual queue depth, version lag, valid/completed tokens and separately synchronized rollout, reward, rescore, update and publication intervals. First run same-distribution controls, then bounded lag/backend mismatches without changing the evaluation budget. Record unsupported/failing paths rather than dropping them. Imported worker contracts, complete backend/gradient equivalence, real queue behavior and distributed throughput remain pending TRN-2 work.
