# Research audit — LLM training and post-training, 2026-11-07

Reviewed the 34-deck syllabus/bibliography and representative L1/L7/L9 labs. The course already covers Muon, DAPO/Dr. GRPO, verifier rewards and evaluation confounds; these are not missing topics. The next addition should connect those ideas to coverage, rollout economics and independently checked experiments. [Repository-wide audit](../../docs/research-audit-2026-11-07.md).

## Primary sources checked

| Source / inspected sections | Finding and limitation |
|---|---|
| [Stanford CS336 Spring 2026](https://cs336.stanford.edu/), assignments/data/scaling; [OLMo-core](https://github.com/allenai/OLMo-core), project overview | Useful syllabus and production-code comparisons. Keep data, optimizer and evaluation contracts tied to revisions. Reading a training repository is not reproducing its published run. |
| Clay et al., [Demystifying Reinforcement Learning Post-Training of Language Models, 2608.24949v1](https://arxiv.org/html/2608.24949v1), §§4–5 and reward/prompt controls | Controlled base priors, reward density and prompt diversity help distinguish coverage from optimization effects. The examples motivate an ablation; they do not prove that RL universally cannot acquire useful new behavior. |
| Zeng et al., [Exploration via Activation-Space Steering for Reinforcement Learning with Verifiable Rewards, 2610.27012v1](https://arxiv.org/html/2610.27012v1), §3, §4.7 and evaluation protocol | A recent exploration proposal builds activation-derived directions for later rollouts. Matched rollout counts are not identical wall-clock cost: inspect replay, bank maintenance and training throughput too. Treat as optional replication reading, not established default practice. |
| [verl author repository](https://github.com/verl-project/verl), rollout/backends and multi-turn links | Shows a production RL stack beyond a local policy update. The old `volcengine/verl` path redirects. Pin the actual backend/configuration and separately identify rollout behavior-policy and learner probabilities before interpreting a policy-ratio experiment. |
| [Google Tuning Playbook](https://github.com/google-research/tuning_playbook), experiment-design section | Tune nuisance hyperparameters under a declared budget. L1 already uses multiple seeds and avoids strong optimizer rankings; retain that caution and extend its reporting protocol. |

## Proposed additions and execution gates

| ID / priority | Targets | Bounded execution / acceptance |
|---|---|---|
| TRN-1 / P1 method | Decks 25–28; L7 `grpo.py`, `task.py` | Add a base-coverage × reward-density ablation to the existing arithmetic policy. Freeze prompt splits and seeds, vary SFT coverage, then compare sparse and documented partial rewards. Record zero-variance groups, accepted reward events, output length and held-out pass@k. Every shaping rule gets an adversarial reward-gaming fixture. |
| TRN-2 / P1 systems bridge | Decks 09/20/26; compute 18/28/29 | Trace one pinned verl rollout → reward → update path. Report behavior-policy version, learner version, log-probability convention, masking, queue depth and rollout/update timing. First prove the ratio is 1 for identical models/masks in the same numerical contract. A one-process source walkthrough is not distributed RL qualification. |
| TRN-3 / P2 recent research | Deck 28; L7 optional target extension | Read ASE's ablations, then compare no steering, random steering and learned steering with fixed data and checkpoint selection. Record activation extraction/replay and search cost. Acceptance: held-out quality plus total time/memory, not only reward improvement per sampled response. No reproduction is claimed here. |
| TRN-4 / P1 reproducibility | Decks 01/32/33; L1/L9 | Add an environment manifest and a selected-configuration reporting worksheet: exact Python/PyTorch/NumPy, data/split/seed, search budget and selection rule. L1/L7 document reference runs on PyTorch 2.15, but a broad `torch 2` install requirement does not recreate that environment. Establish a tested lock in a separate compatibility PR. |
| TRN-5 / retain | Canonical training and preference papers | Keep PPO/DPO/GRPO derivations and current CPU tests. A newer policy method does not invalidate their learning objectives. Compare matched mathematical conventions before asserting one implementation is wrong. |

## Potential projects

**Coverage versus exploration (research replication).** Week 1 freeze base checkpoints and a held-out task split; week 2 implement reward/coverage controls; week 3 run multiple seeds and diagnose sparse no-signal groups; week 4 compare quality, diversity and total rollout cost. A toy positive result is a mechanism demonstration, not frontier reasoning evidence. Open question: which gains survive task transfer and an equal total compute budget?

**Rollout/learner consistency ledger (engineering, research extension possible).** Week 1 annotate one author stack; week 2 build same-model log-probability/mask fixtures; week 3 introduce controlled policy lag; week 4 report ratio/gradient and throughput effects with a fixed evaluation protocol. Full multi-worker conclusions require the actual backend. Compare existing verl facilities before claiming a new correction or scheduler.

No training/benchmark runs or dependency upgrades were performed in this literature audit. Newer papers are proposals to test; their reported improvements are not course reference results.


## Follow-up progress — 2026-11-07

**TRN-1, prepared for review:** decks 26/28 now separate SFT prompt count, measured starting coverage, reward events and held-out strict pass@k. L7 adds a fixed 80/20 split, nested 10/40-prompt SFT interventions, sparse versus deliberately gameable width shaping, independent evaluation, and a complete twelve-run / three-seed CPU ledger. Each reward pair starts from an identical hashed checkpoint; rollout counts, no-signal groups, lengths, diversity and timing boundaries are reported. Six new tests cover controls and counterexamples. This implements a bounded toy ablation, not a frontier-model replication, reliable generalization result or ASE study. The poor held-out result and limitations are retained. TRN-2/3/4 remain separate work.

## TRN-4 manifest/reporting portion — 2026-11-09

The [two-lane evidence note and manifest](cpu_environment_evidence_2026_11.md) records L1's 10 tests on CPython 3.12.14 / PyTorch 2.9.1 without NumPy, and L9's 8 tests on CPython 3.12.14 / NumPy 2.3.5 without Torch. Exact source hashes, commands, warnings, data generator seeds/hashes and fixed selection/search rules are preserved. L1 has 43 shared exact generated examples across its train/validation cohorts, so separate seeds are not represented as strict disjointness. Decks 32/33 now visualize environment/selection evidence; original questions remain in L9. This completes the bounded manifest/reporting portion only: no fresh installation/lock, joint package lane, historical-table reproduction, other-lab, GPU or distributed qualification. Existing lab Python and historical tables are unchanged.
