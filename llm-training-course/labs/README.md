# Labs

CPU-only labs, each checked against a reference value, a closed form or a definition. The
[syllabus](../SYLLABUS.md) lists all ten; labs are added with the part of the course they belong to.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_training_budget/) | A training budget: FLOPs, Chinchilla tokens, days at an MFU, ZeRO and activation memory, pipeline bubbles, batch-size trade-off | 01, 02, 04, 09–11 | Python 3.10+ |
| [L1](L1_pretraining_loop/) | A tiny pre-training loop: hand-written AdamW and Muon, constant, cosine and WSD schedules, QK-norm and z-loss | 01–04 | Python 3.10+, PyTorch 2 (CPU) |
| [L2](L2_data_pipeline/) | A data pipeline: line rules, exact and MinHash dedup, a quality classifier, mixtures and repetition | 05–08 | Python 3.10+ |
| [L3](L3_sft_packing/) | SFT on a tiny model: chat template and response-only loss, packing with a block-diagonal mask, loss averaging across micro-batches | 13–15 | Python 3.10+, PyTorch 2 (CPU) |
| [L4](L4_lora_nf4/) | LoRA from scratch with hand-written gradients and merging, 4-bit NormalFloat, block-wise and double quantization | 16, 17 | Python 3.10+, NumPy |
| [L5](L5_reward_ppo/) | A Bradley-Terry reward model on synthetic preferences, PPO with a KL penalty on a toy policy, overoptimization against a known gold reward | 18–21 | Python 3.10+, PyTorch 2 (CPU) |
| [L6](L6_dpo/) | DPO, IPO and SimPO losses from log-probabilities with gradients; DPO reaches the closed-form RLHF optimum; IPO against DPO on a certain preference | 22–24 | Python 3.10+, NumPy |
| [L7](L7_grpo/) | GRPO and Dr. GRPO on toy addition with a rule-based reward: group advantages, length and difficulty normalizations, the clipped loss | 25–28 | Python 3.10+, PyTorch 2 (CPU) |

Run every lab's checks from the lab directory with `python -m unittest`.
