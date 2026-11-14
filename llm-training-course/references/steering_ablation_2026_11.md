# Training-time steering: frozen toy ablation — 2026-11-14 UTC

This is the bounded TRN-3 CPU teaching experiment, not an ASE reproduction. All nine runs are retained. Held-out success is poor; learned steering does not win every seed. Neither pooled counts nor these short timings establish a general quality or speed ranking.

## Fixed design and selection

The [protocol](../labs/L7_grpo/steering_protocol.json) was committed at `2b53e38` before execution. The [runner](../labs/L7_grpo/steering_ablation.py) and eight controls were committed at `145d931`. All final checkpoints were committed at `4d5e08c` before held-out evaluation. No tuning, best-seed selection, refit or checkpoint selection followed evaluation. The fixed final step is 80.

The existing L7 autoregressive MLP has hidden width 64, digit/filler/EOS vocabulary and maximum length 8; it is not a Transformer. Training uses 73 ordered operand pairs (40 unordered groups); held-out uses 27 (15 groups), keeping reversed operands together. The fixed 24 SFT prompts are a training subset. Every arm per seed starts from the same verified parameter hash after 300 SFT steps. Seeds are 11, 23 and 37. Each run uses 80 RL steps × 8 prompts × 6 responses = 3,840 responses / 30,720 sampled token slots, with two Adam updates per batch. Sampling continues through all slots; valid-loss masks include EOS. Task reward requires EOS and exactly the correct sum after removing filler tokens.

The three arms are no steering, random and learned. Both steering arms replay real within-prompt success/failure pairs without intervention, read post-Tanh activations at the quarter-position, fit PCA4 and deterministic two-cluster means in the original hidden space, then cosine-match and EMA-smooth the bank (beta .5). Random replaces candidate directions with Gaussian directions of matching norms before the same matching/EMA. Routes use two clean and two responses per direction, alpha .5, after bank activation. The minimum buffer is four pairs, capacity128, bank interval2. Homogeneous groups and degenerate clusters do not fabricate a direction. Bank publication follows both local actor updates; this is not distributed weight synchronization.

The learner and old anchor are unsteered; behavior probabilities reflect the actual routed proposal. The masked token-mean clipped GRPO loss uses detached `min(2, exp(old - behavior))` correction and `exp(new - old)` PPO ratio, clip .2, without batch weight normalization. Sampler/data RNGs are separated from bank/random-direction RNGs. Equal sampled-response counts do not equalize replay work or total compute. There is one frozen configuration per arm, no hyperparameter search.

## All held-out results and measured cost

Each final checkpoint was evaluated in a fresh process on the same 27 prompts × 8 samples, steering disabled, common sampling seed617. The empirical prompt success column counts prompts with at least one success among those eight draws; it is not a confidence interval. The same prompts repeat across seeds.

| Seed | Arm | Correct responses | Successful prompts | Replayed trajectories | Bank updates | Train phase seconds | Peak retained bank/buffer bytes | Raw training record |
|---|---|---|---|---|---|---|---|---|
| 11 | learned | 3/216 | 1/27 | 414 | 21 | 0.520103 | 10752 | [record](../labs/L7_grpo/steering_runs/seed11-learned/train.json) |
| 11 | none | 0/216 | 0/27 | 0 | 0 | 0.512053 | 0 | [record](../labs/L7_grpo/steering_runs/seed11-none/train.json) |
| 11 | random | 1/216 | 1/27 | 364 | 14 | 0.517473 | 20480 | [record](../labs/L7_grpo/steering_runs/seed11-random/train.json) |
| 23 | learned | 1/216 | 1/27 | 300 | 19 | 0.510604 | 9728 | [record](../labs/L7_grpo/steering_runs/seed23-learned/train.json) |
| 23 | none | 1/216 | 1/27 | 0 | 0 | 0.490910 | 0 | [record](../labs/L7_grpo/steering_runs/seed23-none/train.json) |
| 23 | random | 0/216 | 0/27 | 268 | 14 | 0.513637 | 15872 | [record](../labs/L7_grpo/steering_runs/seed23-random/train.json) |
| 37 | learned | 0/216 | 0/27 | 234 | 14 | 0.507825 | 7680 | [record](../labs/L7_grpo/steering_runs/seed37-learned/train.json) |
| 37 | none | 0/216 | 0/27 | 0 | 0 | 0.505216 | 0 | [record](../labs/L7_grpo/steering_runs/seed37-none/train.json) |
| 37 | random | 1/216 | 1/27 | 276 | 14 | 0.508585 | 10752 | [record](../labs/L7_grpo/steering_runs/seed37-random/train.json) |

Totals are 1/648 no steering, 2/648 random and 4/648 learned; successful-prompt counts are 1/81, 2/81 and 2/81. These tiny counts and three fixed seeds cannot establish superiority, significance or transfer. We did not revise the scorer after seeing results. A separate decoding calculation independently checked all 1,944 raw responses and every stored reward/aggregate against strict EOS/filler/digit rules.

[Summary and evaluation process ledger](steering_ablation_summary_2026_11.json), [pre-evaluation checkpoint/validation manifest](steering_checkpoint_manifest_2026_11.json). Each run directory contains `final.pt` (course-owned tensor weights), `train.json` and `evaluation.json`, including raw generated tokens. Checkpoint file and parameter hashes are verified before evaluation; loading uses `weights_only=True`. The JSON summary links every raw report; the protocol and runner hashes remain unchanged.

Timing boundaries matter: train phase includes SFT, sampling, scoring, replay, bank fitting and updates, but excludes interpreter/Torch startup. The process ledgers include fresh-process elapsed time; evaluation records include sample and phase time. Per-phase times, valid/replayed tokens, fit attempts and publication/steering traces are in each training record. Peak retained bank/buffer bytes counts those persistent tensors only, excluding temporary SVD/optimizer/model tensors. Full-worker peak RSS is also recorded (macOS bytes), including imported libraries and allocator history; it is not GPU memory or a per-phase increment. Short CPU timings are descriptive, not a throughput ranking.

Runtime: CPython3.12.14, Torch2.9.1, NumPy2.3.5, macOS27.0.1 arm64, CPU one thread, using the [qualified joint lock](tested_cpu_lock_2026_11.md). No GPU/distributed/backend experiment ran. The original policy, task, GRPO implementation and historical coverage tables remain unchanged.

## Primary-source reading and departures

Zeng et al., [Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards, v1](https://arxiv.org/html/2610.27012v1), §§3, 4.4–4.7 and Appendices B–C, inspected 2026-11-14 UTC. Reading focused on within-problem pairing, clean replay, clustered activation contrasts, random-direction control, stabilized routing, behavior correction and evaluation with intervention disabled. No upstream figure was copied; slide9 contains an original table diagram.

The paper uses Qwen3/DAPO/verl/vLLM/GPU math training; this is a much smaller MLP adaptation (K2/PCA4 rather than K3/PCA64), not a test of its frontier benchmark or scaling/speed claims. AppendixB describes MATH500-based test-adaptive checkpoint selection; our final-step rule is fixed before held-out scoring. AppendixC specifies capped token-level importance sampling without batch normalization. No paper benchmark number is reproduced here. Generalization across disjoint operands is still elementary arithmetic, not out-of-distribution reasoning. Real model/task transfer and equal-total-compute studies remain open.

## Controls and reproduction

Eight new controls pass: zero-intervention equivalence, exception-safe restoration, within-prompt pairing, balanced/normed routes, full-space clustered means/degenerate rejection, cosine matching before EMA, literal correction loss/gradients and detached behavior, and split/SFT isolation. The whole L7 suite passes 28 tests. Raw validation logs and command outputs are retained in the checkpoint manifest.

From `labs/L7_grpo`, using the qualified interpreter:

```bash
python -m unittest discover -v
python steering_ablation.py train --condition learned --seed 11 --directory /tmp/new-seed11-learned
# Preserve and freeze the checkpoint before scoring; use a NEW directory for each arm/seed.
python steering_ablation.py evaluate --directory /tmp/new-seed11-learned
```

The runner refuses existing training directories or overwriting evaluation records. It checks selected package versions; recorded hashes identify the source/protocol, rather than promising cross-platform determinism. The frozen published results must not be overwritten.
