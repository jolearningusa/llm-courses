# Lab L7: GRPO with a verifiable reward

Used with decks 25 to 28. CPU only, Python 3.10+ and PyTorch 2. A full run takes about 10 seconds.

RL for reasoning replaces the reward model with a program that checks the answer, and PPO's value model
with a baseline computed from a group of answers to the same prompt. This lab trains a tiny autoregressive
policy to add two digits with GRPO, then with Dr. GRPO's corrections, from the same starting point.

| File | What it holds | Deck |
|---|---|---|
| `task.py` | Prompts a + b, a 12-token vocabulary with a meaningless filler token, and the rule-based reward | 25, 27 |
| `policy.py` | A small autoregressive policy: per-token log-probabilities with an end-of-response mask, and sampling | 26 |
| `grpo.py` | Group advantages, GRPO's and Dr. GRPO's token weights, the clipped loss, an SFT warm-up on 30 prompts, and the RL loop | 26, 28 |

## Run

```bash
python -m unittest test_grpo.py   # about 3 seconds
python grpo.py                    # SFT warm-up, then GRPO and Dr. GRPO for 300 steps each
python grpo.py --seed 1 --steps 600
```

## Checks

- The reward is 1 only for a finished response whose digits spell a + b; filler tokens are ignored.
- Group advantages sum to zero in every group, and a group whose answers are all right (or all wrong) gives
  no signal at all. That is why DAPO filters such groups out (deck 28).
- Dividing by the group's standard deviation gives a rare success a larger advantage than a common one,
  and gives every group the same spread: the question-level difficulty bias of Liu et al.
  ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783)). Dr. GRPO keeps r − mean.
- GRPO's per-response averaging gives each token of a 2-token answer four times the weight of a token in an
  8-token answer, so a long wrong answer is penalized less per token. Dr. GRPO weights every token the same.
- The clipped loss has no gradient once the ratio passes 1 + ε for a positive advantage.
- 150 steps of GRPO raise the mean reward over all 100 prompts by more than 0.1 above the SFT start.

## Reference results

`python grpo.py` on our machine (PyTorch 2.15, CPU):

```text
after SFT on 30 prompts: mean reward 0.273 (length: correct 3.93, incorrect 4.80)
     GRPO after 300 steps: mean reward 0.506 (length: correct 2.43, incorrect 2.79)
 Dr. GRPO after 300 steps: mean reward 0.489 (length: correct 2.46, incorrect 2.49)
```

Seeds 1 and 2 reach 0.42 and 0.40 from about 0.28, with both methods within 0.02 of each other. RL also
removes most filler tokens, because they add no reward. At this scale the length bias is too small to
separate the two methods reliably across seeds; task 2 makes it visible.

## Tasks

1. Add the KL penalty to the reference policy with GRPO's estimator π<sub>ref</sub>/π − log(π<sub>ref</sub>/π) − 1
   (deck 26). How does the mean reward after 300 steps change for coefficients 0.01, 0.04 and 0.2?
2. Make the length bias visible: give the filler token a small positive logit bonus at the start, raise
   `MAX_LEN` to 32, and train on hard prompts only (sums above 12). Track the length of incorrect responses
   under GRPO and Dr. GRPO.
3. Replace the group mean with the leave-one-out mean of RLOO (deck 26). Show that the advantages equal
   G/(G − 1) times Dr. GRPO's.
4. Implement DAPO's dynamic sampling: drop groups whose rewards are all equal and sample more prompts until
   the batch is full. How many more samples does a step need early and late in training?
5. Measure pass@8 and majority voting over 8 samples before and after RL. DeepSeekMath found that RL raised
   Maj@K but not Pass@K (deck 26). Does the toy agree?
6. Give partial credit (0.5 for the right number of digits). Does the policy learn to game it?


## Controlled coverage × reward experiment

Deck 26 slide 9 and deck 28 slide 8 use this extension. It keeps the original `grpo.py` demo and its historical results above. **That demo evaluates all 100 prompts used by RL; it is not held-out generalization.** The new run uses an explicit independent 20-prompt evaluation set.

```bash
python -m unittest test_grpo.py test_coverage.py
python coverage_ablation.py --output /tmp/coverage-new.json
```

`coverage_ablation.py` uses PyTorch on CPU only, with no NumPy calls, downloads, API calls or external checkpoints. It refuses to overwrite an existing file. The checked-in [complete reference ledger](coverage_reference.json) was executed on 2026-11-07 with Python 3.12.14 / PyTorch 2.9.1 / macOS arm64, one CPU thread and float32. PyTorch warned that NumPy was absent; no NumPy conversion was used. This verifies this extension on that environment; it does not qualify the whole course on every PyTorch 2 version or supersede the historical 2.15 demo.

### Freeze the controls before training

| Control | Declared setting |
|---|---|
| Prompt split | Shuffle all 100 ordered digit pairs with seed 314159: first 80 are RL/SFT pool, final 20 are held out from both training stages |
| SFT intervention | Nested first 10 or first 40 pool prompts; same 300 steps × 32 examples, Adam learning rate 0.003 and filler distribution |
| Measured base coverage | Strict sampled pass@1/4/8 on **both** the 80-prompt pool and 20 held-out prompts before RL; SFT set size alone is not coverage probability |
| Pairing | Seeds 0, 1, 2; each reward arm clones the exact same SFT state (SHA-256 stored); common prompt/sampling seed within each pair |
| RL controls | Existing GRPO, G=8, 16 prompts per step, 150 steps, two inner updates, learning rate 0.003, std/response-length normalization, no KL; fixed final checkpoint, no search/selection |
| Evaluation | 16 stochastic responses per held-out prompt, seed 10000 + trial seed, strict independent checker; average `1 − C(n−c,k)/C(n,k)` per prompt for k=1/4/8 (L9 explains the estimator) |

The held-out pairs can share an operand, sum or reversed pair with training; this is a tiny compositional holdout, not an unseen mathematical domain. SFT size changes exposure/repetition as well as nominal coverage. The simple embedding policy need not generalize addition; measure its actual starting success probability. Zero successes in 16 samples is finite evidence, not proof of zero support. Common evaluation seeds support paired comparisons, not independent confidence intervals.

### The two reward arms and their attack fixtures

- **Sparse:** existing exact-answer reward, 1 only for the right number followed by EOS. Wrong number, missing EOS and empty answer tests remain in the suite; the independent evaluation checker also checks them.
- **Width proxy:** exact=1; otherwise 0.5 for a finished response with the same digit count as the answer; 0 otherwise. This deliberately implements task 6's weak shaping. For `1+2`, `9<EOS>` receives 0.5 but is wrong; for `7+8`, `16<EOS>` does too. Missing EOS gets zero. These attacks are executable tests. A higher proxy reward or positive-event frequency does not validate arithmetic. It is not a PRM or a trusted dense correctness reward.

`train(..., reward_fn=..., trace=...)` preserves the default algorithm. Each step logs total groups, equal-reward groups, positive reward events, strict successes, reward sum, response count, valid tokens, executed sampling slots and distinct decoded answers per group. No groups are discarded or resampled. Equal rewards give zero relative advantages; this means no **direct advantage signal**, not necessarily zero parameter movement after Adam has accumulated momentum. Positive reward events are not distinct discoveries, and filler variation is not reasoning diversity.

All arms sample 19,200 responses and execute 153,600 token slots (the sampler always runs 8 positions, including post-EOS slots). Logical output lengths include the first EOS and exclude padding. Record both; equal response counts do not prove equal total compute. The ledger stores SFT/RL time separately; SFT is reused across reward arms. Evaluation, hashing and file writing are outside RL timing; training timing includes trace collection. Cross-machine timings and these sub-second CPU trials are not throughput benchmarks.

### Read the result before claiming a gain

The slide table is generated from the checked-in ledger, averaging the three seeds without selecting a winner. Every seed and per-prompt strict count remains in that file. Report seed ranges and paired differences too; these seeds share the same 20 test prompts, so the range is not a confidence interval. Width shaping gives many wrong answers positive reward and reduces equal-reward groups, but strict held-out success remains low. Neither shaping nor more SFT prompts establishes reliable generalization here.

Run a larger predeclared seed/split study before testing transfer. Keep hyperparameter/search budgets equal, freeze checkpoint selection without test scores, and evaluate a genuinely different task distribution. This toy does not test activation steering, reproduce Clay et al.'s LLM/PRM results, or settle whether RL learns new reasoning.

### Original deck closing questions

The revised deck 26 closing slide links these derivations here: show group advantages sum to zero; derive the bias of a group mean containing its own sample versus RLOO; interpret the paper's Maj@K versus Pass@K result within its cohort; explain when clipping becomes active. Deck 28's inference-compute and length-reward links remain in decks 27/32 and tasks 1–6 above.


### Executed seed-level outcomes

Percentages below are from the complete ledger, not chosen checkpoints. Positive events use the **training reward**; strict pass@k uses held-out correctness. Different populations and denominators are intentional.

| Seed | SFT prompts | Reward | Held-out pass@1, before → after | Held-out pass@8, before → after | Positive training events | Equal-reward groups |
|---|---|---|---|---|---|---|
| 0 | 10 | sparse | 5.62% → 0.31% | 13.83% → 2.50% | 22.2% | 91.2% |
| 0 | 10 | width_proxy | 5.62% → 0.31% | 13.83% → 2.50% | 91.8% | 76.9% |
| 0 | 40 | sparse | 0.94% → 0.00% | 6.33% → 0.00% | 49.6% | 81.0% |
| 0 | 40 | width_proxy | 0.94% → 0.00% | 6.33% → 0.00% | 95.6% | 72.2% |
| 1 | 10 | sparse | 1.88% → 0.00% | 8.64% → 0.00% | 18.9% | 94.3% |
| 1 | 10 | width_proxy | 1.88% → 0.00% | 8.64% → 0.00% | 90.2% | 75.8% |
| 1 | 40 | sparse | 0.31% → 0.00% | 2.50% → 0.00% | 49.8% | 86.3% |
| 1 | 40 | width_proxy | 0.31% → 0.31% | 2.50% → 2.50% | 96.7% | 74.2% |
| 2 | 10 | sparse | 1.25% → 0.31% | 7.00% → 2.50% | 22.9% | 88.2% |
| 2 | 10 | width_proxy | 1.25% → 0.00% | 7.00% → 0.00% | 89.9% | 75.8% |
| 2 | 40 | sparse | 1.88% → 2.50% | 7.44% → 12.44% | 51.3% | 80.6% |
| 2 | 40 | width_proxy | 1.88% → 1.56% | 7.44% → 4.94% | 95.1% | 72.9% |

Across the three seeds, final held-out pass@1 ranges from 0–0.31% (10/sparse), 0–0.31% (10/width), 0–2.50% (40/sparse), and 0–1.56% (40/width). These ranges describe this run, not statistical significance.

## Optional rollout / learner consistency

Use the [pinned verl reading and CPU evidence note](../../references/rollout_consistency_2026_11.md) with deck 20's closing slides. This adds probability-contract controls without changing `policy.py`, `grpo.py`, earlier ablations or their tests.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python rollout_consistency.py --output rollout-new.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  test_rollout_consistency test_grpo.TestReward test_grpo.TestGroupAdvantages \
  test_grpo.TestLengthNormalization test_coverage -v
```

Use a fresh output path. The [actual report](rollout_consistency_report.json) records sampled responses, observed sampling logits as selected log-probs, independent rescoring, masks, three policy identities, strict rewards, a synthetic-advantage update and single CPU phase intervals. Nineteen selected tests pass (eight new); the longer `TestTraining` case and historical tables are not rerun by this command.

1. Draw `q` (rollout), `p_old` (fixed anchor), `p_theta` (current learner). Identify the denominator of PPO's ratio separately from decoupled rollout correction. A KL-reference policy is not automatically either denominator.
2. Predict when identical weights can yield a ratio other than one: different temperature/filters, tokens/shift, mask or backend/precision. Reproduce the temperature and previous-token negative controls. Check the mask independently; ratio one alone does not detect a wrongly excluded EOS.
3. Compare token ratios `[2,3]`, their sequence product `6`, and the exponential of mean log-ratio. Explain why reductions cannot be silently exchanged.
4. Explain why the four zero strict rewards cannot supply the signed advantages used for the isolated update. Inspect the changed learner hash and unchanged rollout/anchor hashes without calling it a quality gain.
5. Design a real-backend ledger: sample/weight versions, valid masks, queue depth, completed work and stage timing. The local synchronous batch counts and CPU intervals are not Ray/vLLM throughput or async evidence.

The observer and fingerprints are reproduction aids, not authenticated runtime attestations. The new finite-ratio helper's padding control is scoped to that helper; no blanket nonfinite-input robustness claim is made for the original policy loss. Retained deck 20 questions: helpfulness/safety reward separation, rating margins, KL versus early stopping, and comparing alignment-tax/bonus study setups. Historical questions and prose also remain in slide notes.

## Optional: training-time steering with a frozen budget

The [TRN-3 study](../../references/steering_ablation_2026_11.md) implements no/random/learned activation steering, freezes all final checkpoints before evaluation and records all nine runs. Held-out generalization is poor; inspect the per-seed outcomes and replay/time/memory costs before claiming a gain. This separate experiment does not change the coverage reference table above. See the evidence note for protocol, controls, raw outputs and commands.
