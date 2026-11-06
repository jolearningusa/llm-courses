# Syllabus: LLM Training and Post-Training

Fifteen weeks, two or three paper decks a week, with one lab session every week or two. The course follows
one model from raw web text to a chat and reasoning model: pre-training (recipe, data, scale, long context),
then post-training (supervised fine-tuning, parameter-efficient tuning, RLHF, direct preference methods,
RL for reasoning, distillation), then safety tuning and evaluation. It also splits into two shorter
courses: **pre-training** (Parts 0 to IV, weeks 1 to 6) and **post-training** (Parts V to XI, weeks 7 to
15). The post-training half needs only deck 01 and lab L1 from the first half.

Each paper deck follows five steps: the **problem** the authors faced, the **method** (the objective, data
or recipe, with the equation that defines it), the **evidence** in the paper, the **training impact**
(quality, compute, data, memory, stability, what it made possible), and the **afterlife** (which later
reports kept, changed or dropped it, checked against their text).

Architecture is taught in [LLM Transformer Architecture](../llm-transformer-architecture-course/), and
the systems side of training (kernels, collectives, sharding, checkpointing) in
[Efficient LLM Computing Systems](../llm-compute-systems-course/). Where a topic is already a deck there,
this course links to it instead of repeating it (see "Decks this course links to" below).

## Prerequisites

Python, NumPy, linear algebra, probability, and the basics of training a neural network: a loss, backpropagation,
stochastic gradient descent and what a softmax and cross-entropy do. Labs L1, L3, L5 and L7 also use PyTorch.
Classic architectures (MLPs, CNNs, RNNs and LSTMs) and the basics of training them are taught in
[ML 101: Neural Network Foundations](../ml101-neural-networks-course/), not here. You should also
know what a decoder-only Transformer computes, at the level of decks 01 and 02 of the architecture course.

## Parts and decks

| Part | Deck | Paper(s) |
|---|---|---|
| 0 Map | 00 | Course map |
| I Pre-training recipe | 01 | The training loop: next-token loss, Adam and AdamW, warm-up, cosine and WSD schedules (MiniCPM, Hägele et al.) |
| | 02 | Batch size and hyperparameter transfer: the critical batch size, μP |
| | 03 | Training stability: loss spikes, z-loss, QK-norm, small-scale proxies (PaLM, Wortsman et al., OLMo 2) |
| | 04 | Muon and optimizers beyond AdamW (Jordan, Moonlight) |
| II Pre-training data | 05 | Web corpora: CCNet, The Pile, RefinedWeb, Dolma |
| | 06 | Deduplication and quality filtering: Lee et al., FineWeb and FineWeb-Edu, DataComp-LM |
| | 07 | Data mixtures and repetition: DoReMi, RegMix, data-constrained scaling |
| | 08 | Code, math and synthetic data, and mid-training: phi-1, the DeepSeekMath corpus, annealing |
| III Training at scale | 09 | Parallelism for model builders: ZeRO, Megatron-LM, GPipe, 3D parallelism (links to the compute course) |
| | 10 | Low-precision training: mixed precision, FP8 formats, DeepSeek-V3's FP8 recipe |
| IV Long context and continued pre-training | 11 | Long-context continued pre-training: Xiong et al., Fu et al., ProLong, the Llama 3 stages |
| | 12 | Domain-adaptive and continual pre-training: Don't Stop Pretraining, Ibrahim et al. |
| V Supervised fine-tuning | 13 | Instruction tuning: FLAN, T0, Flan-T5 and the Flan Collection |
| | 14 | Instruction data: Self-Instruct, Alpaca, LIMA |
| | 15 | An open post-training recipe end to end: Tülu 3, with SFT loss masking and packing |
| VI Parameter-efficient tuning | 16 | Adapters, prefix tuning and LoRA |
| | 17 | QLoRA, DoRA, and what LoRA learns and forgets |
| VII Reward models and RLHF | 18 | Learning from human preferences: Christiano et al., Ziegler et al., summarization from feedback, Bradley-Terry reward models |
| | 19 | PPO and InstructGPT: clipped objective, GAE, the KL penalty |
| | 20 | RLHF at scale: Anthropic HH, Llama 2-Chat, reward-model overoptimization |
| | 21 | AI feedback: Constitutional AI and RLAIF |
| VIII Direct preference methods | 22 | DPO |
| | 23 | DPO variants: IPO, KTO, ORPO, SimPO |
| | 24 | Online and iterative preference learning: Self-Rewarding LMs, DPO against PPO |
| IX RL for reasoning | 25 | Verifiers and process rewards: GSM8K verifiers, STaR, Let's Verify Step by Step, Math-Shepherd |
| | 26 | GRPO and REINFORCE baselines: DeepSeekMath, RLOO |
| | 27 | RL with verifiable rewards: DeepSeek-R1, Kimi k1.5, Tülu 3 RLVR |
| | 28 | Fixing GRPO and spending test-time compute: DAPO, Dr. GRPO, Snell et al. |
| X Distillation | 29 | Knowledge distillation: Hinton et al., sequence-level KD, DistilBERT |
| | 30 | Distilling LLMs: MiniLLM, on-policy distillation (GKD), Gemma 2, R1's distilled models |
| XI Safety and evaluation | 31 | Safety tuning: red teaming, Safety-Tuned LLaMAs, fine-tuning attacks, GCG, deliberative alignment |
| | 32 | Evaluating LLMs: MMLU, HumanEval and pass@k, HELM, MT-Bench and Chatbot Arena, contamination |
| | 33 | Capstone: plan a training run from base model to chat model |

## Decks this course links to

These topics already have a deck in a sibling course. The deck here gives the training view and links there.

| Topic | Deck in this course | Linked deck |
|---|---|---|
| Masked LM and span-corruption objectives | 01 | Architecture [03 BERT](../llm-transformer-architecture-course/slides/03_bert.html), [04 T5](../llm-transformer-architecture-course/slides/04_t5.html) |
| Kaplan scaling laws, compute-optimal tokens | 01, 07 | Architecture [05 GPT-3 and scaling laws](../llm-transformer-architecture-course/slides/05_gpt3_scaling_laws.html), [06 Chinchilla](../llm-transformer-architecture-course/slides/06_chinchilla.html) |
| QK-norm and normalization placement | 03 | Architecture [17 Pre-LN and RMSNorm](../llm-transformer-architecture-course/slides/17_preln_rmsnorm.html), [30 Qwen](../llm-transformer-architecture-course/slides/30_qwen.html) |
| Multi-token prediction as an extra loss | 01 | Architecture [27 Multi-token prediction](../llm-transformer-architecture-course/slides/27_multi_token_prediction.html) |
| Data quality at small scale (Phi-3, OLMo) | 08 | Architecture [33 Phi-3, OLMo, Kimi K2 and gpt-oss](../llm-transformer-architecture-course/slides/33_phi_olmo_kimi_gptoss.html) |
| DDP, ZeRO, FSDP, collectives, tensor and pipeline parallelism | 09 | Compute [25](../llm-compute-systems-course/slides/25_ddp_zero_fsdp.html), [26](../llm-compute-systems-course/slides/26_collectives_overlap.html), [27](../llm-compute-systems-course/slides/27_parallelism_recomputation.html), [29 checkpointing](../llm-compute-systems-course/slides/29_checkpoint_recovery.html) |
| Number formats and numerics | 10 | Compute [13 Precision and numerics](../llm-compute-systems-course/slides/13_precision_numerics.html) |
| Position Interpolation and YaRN | 11 | Architecture [10 Position Interpolation and YaRN](../llm-transformer-architecture-course/slides/10_pi_yarn.html) |
| Chat templates, loss masks and packing | 15 | Architecture [11 From conversation to tokens](../llm-transformer-architecture-course/slides/11_conversation_to_tokens.html) |
| Llama 2-Chat | 20 | Architecture [28 LLaMA and Llama 2](../llm-transformer-architecture-course/slides/28_llama_1_2.html) |
| DeepSeek-R1, R1-Zero | 27 | Architecture [31 DeepSeek-V3 and R1](../llm-transformer-architecture-course/slides/31_deepseek_v3_r1.html) |
| Gemma 2 distillation | 30 | Architecture [32 Gemma 1, 2 and 3](../llm-transformer-architecture-course/slides/32_gemma.html) |

## Weekly plan

| Week | Decks | Lab |
|---|---|---|
| 1 | 00, 01, 02 | L0: a training budget: FLOPs, tokens, and memory per GPU |
| 2 | 03, 04 | L1: a tiny pre-training loop with AdamW and three schedules |
| 3 | 05, 06 | L2: a data pipeline: exact and MinHash dedup, a quality classifier |
| 4 | 07, 08 | L2 (continued): temperature-sampled mixtures |
| 5 | 09, 10 | Extends L0: ZeRO stages and activation memory |
| 6 | 11, 12 | Paper critique 1 |
| 7 | 13, 14, 15 | L3: SFT with response-only loss and packing |
| 8 | 16, 17 | L4: LoRA from scratch and 4-bit NormalFloat |
| 9 | 18, 19 | L5: a Bradley-Terry reward model and PPO on a toy policy |
| 10 | 20, 21 | L5 (continued): the KL penalty and overoptimization |
| 11 | 22, 23, 24 | L6: DPO, IPO and SimPO losses from log-probabilities |
| 12 | 25, 26, 27 | L7: GRPO with a verifiable reward |
| 13 | 28 | Paper critique 2 |
| 14 | 29, 30, 31 | L8: forward and reverse KL, on-policy distillation |
| 15 | 32, 33 | L9: pass@k, ratings from pairwise votes, contamination; capstone presentations |

## Labs

All labs run on a laptop CPU with NumPy or PyTorch and tiny models or toy distributions. Each lab checks its
output against a reference implementation, a closed form or a value computed by hand.

| Lab | Task | Check |
|---|---|---|
| L0 | Compute the FLOPs, Chinchilla-optimal tokens and per-GPU memory (weights, gradients, Adam states, ZeRO stages 1 to 3) of a training run | Matches the 6ND estimate and the memory table in the ZeRO paper |
| L1 | Train a small character-level model with a hand-written AdamW and warm-up-cosine, WSD and constant schedules | Hand-written AdamW equals `torch.optim.AdamW`; fixed seeds and budget, loss reported with spread across seeds |
| L2 | Exact and MinHash near-duplicate removal, an n-gram quality classifier and temperature-sampled mixtures | MinHash estimate tracks exact Jaccard similarity; sampled proportions match the target mixture |
| L3 | Supervised fine-tuning of a tiny model with the loss on assistant tokens only, with packed sequences | Masked tokens get zero gradient; packed and unpacked batches give the same loss |
| L4 | A LoRA layer, merging it into the base weight, and 4-bit NormalFloat quantization | Merged and unmerged outputs are equal; trainable-parameter counts match the formula; NF4 error below 4-bit integer error on normal weights |
| L5 | A Bradley-Terry reward model on synthetic preferences, then PPO with a KL penalty on a toy policy | The reward model ranks held-out pairs correctly; the clipped objective stops large policy steps |
| L6 | DPO, IPO and SimPO losses computed from log-probabilities on a toy policy | DPO's optimum equals the closed-form KL-regularized RLHF policy |
| L7 | GRPO on a toy arithmetic task with a rule-based reward, with and without Dr. GRPO's corrections | Group advantages sum to zero; accuracy rises above the starting policy |
| L8 | Forward and reverse KL distillation of a two-mode teacher, then on-policy distillation | Forward KL covers both modes, reverse KL picks one |
| L9 | The unbiased pass@k estimator, Bradley-Terry ratings from pairwise votes, and an n-gram contamination check | pass@k matches brute-force enumeration; ratings recover the generating strengths |

## Assessment

Labs 50%, two paper critiques 20%, capstone 30%.

- **Paper critique.** Pick a training or post-training paper not covered in class and write two pages in the
  five-step format, including which deck's method it changes and one claim you checked against its tables.
- **Capstone (deck 33).** Plan the training of a chat and reasoning model for a given compute, data and
  memory budget: the token budget and data mixture, the parallelism and precision, the SFT, preference and RL
  stages, and the evaluations that decide when each stage is done. Defend each choice with the papers in this
  course and compute the budget with the L0 tools.
