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
