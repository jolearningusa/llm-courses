# Bibliography

Every paper the course reads, grouped by the part of the course that reads it. For the original entries, on 2026-11-06 each arXiv
entry's ID, title and first author was checked on its arXiv abstract page, and each blog post on its own
page. The year is that of arXiv v1, or of publication for sources without an arXiv paper. Decks cite
sections and tables of these papers. Papers that are already a deck in the
[architecture course](../../llm-transformer-architecture-course/references/bibliography.md) (BERT, T5, the Kaplan
scaling laws, Chinchilla, multi-token prediction) are cited from there and not repeated here. The list holds 98 distinct sources. Dated supplements below record their own verification dates.

## Part I. The pre-training recipe

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 01 | Adam: A Method for Stochastic Optimization | Kingma, 2014 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| 01 | Decoupled Weight Decay Regularization (AdamW) | Loshchilov, 2017 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| 01 | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies (WSD schedule) | Hu, 2024 | [arXiv:2404.06395](https://arxiv.org/abs/2404.06395) |
| 01 | Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations | Hägele, 2024 | [arXiv:2405.18392](https://arxiv.org/abs/2405.18392) |
| 02 | An Empirical Model of Large-Batch Training | McCandlish, 2018 | [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) |
| 02 | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (μP) | Yang, 2022 | [arXiv:2203.03466](https://arxiv.org/abs/2203.03466) |
| 03 | PaLM: Scaling Language Modeling with Pathways | Chowdhery, 2022 | [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) |
| 03 | Small-scale proxies for large-scale Transformer training instabilities | Wortsman, 2023 | [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) |
| 03 | 2 OLMo 2 Furious | Team OLMo, 2024 | [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) |
| 04 | Muon: An optimizer for hidden layers in neural networks | Jordan, 2024 | [Blog post](https://kellerjordan.github.io/posts/muon/) |
| 04 | Muon is Scalable for LLM Training (Moonlight) | Liu, 2025 | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) |

## Part II. Pre-training data

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 05 | CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data | Wenzek, 2019 | [arXiv:1911.00359](https://arxiv.org/abs/1911.00359) |
| 05 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling | Gao, 2020 | [arXiv:2101.00027](https://arxiv.org/abs/2101.00027) |
| 05 | The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only | Penedo, 2023 | [arXiv:2306.01116](https://arxiv.org/abs/2306.01116) |
| 05 | Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research | Soldaini, 2024 | [arXiv:2402.00159](https://arxiv.org/abs/2402.00159) |
| 06 | Deduplicating Training Data Makes Language Models Better | Lee, 2021 | [arXiv:2107.06499](https://arxiv.org/abs/2107.06499) |
| 06 | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale | Penedo, 2024 | [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) |
| 06 | DataComp-LM: In search of the next generation of training sets for language models | Li, 2024 | [arXiv:2406.11794](https://arxiv.org/abs/2406.11794) |
| 07 | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining | Xie, 2023 | [arXiv:2305.10429](https://arxiv.org/abs/2305.10429) |
| 07 | RegMix: Data Mixture as Regression for Language Model Pre-training | Liu, 2024 | [arXiv:2407.01492](https://arxiv.org/abs/2407.01492) |
| 07 | Scaling Data-Constrained Language Models | Muennighoff, 2023 | [arXiv:2305.16264](https://arxiv.org/abs/2305.16264) |
| 08 | Textbooks Are All You Need (phi-1) | Gunasekar, 2023 | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) |
| 08 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Shao, 2024 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| 08 | The Llama 3 Herd of Models (annealing, long-context stages) | Grattafiori, 2024 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |

## Part III. Training at scale

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 09 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari, 2019 | [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) |
| 09 | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | Shoeybi, 2019 | [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) |
| 09 | GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism | Huang, 2018 | [arXiv:1811.06965](https://arxiv.org/abs/1811.06965) |
| 09 | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | Narayanan, 2021 | [arXiv:2104.04473](https://arxiv.org/abs/2104.04473) |
| 09 | Reducing Activation Recomputation in Large Transformer Models | Korthikanti, 2022 | [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) |
| 10 | Mixed Precision Training | Micikevicius, 2017 | [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) |
| 10 | FP8 Formats for Deep Learning | Micikevicius, 2022 | [arXiv:2209.05433](https://arxiv.org/abs/2209.05433) |
| 10 | DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |

## Part IV. Long context and continued pre-training

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 11 | Effective Long-Context Scaling of Foundation Models | Xiong, 2023 | [arXiv:2309.16039](https://arxiv.org/abs/2309.16039) |
| 11 | Data Engineering for Scaling Language Models to 128K Context | Fu, 2024 | [arXiv:2402.10171](https://arxiv.org/abs/2402.10171) |
| 11 | How to Train Long-Context Language Models (Effectively) (ProLong) | Gao, 2024 | [arXiv:2410.02660](https://arxiv.org/abs/2410.02660) |
| 12 | Don't Stop Pretraining: Adapt Language Models to Domains and Tasks | Gururangan, 2020 | [arXiv:2004.10964](https://arxiv.org/abs/2004.10964) |
| 12 | Simple and Scalable Strategies to Continually Pre-train Large Language Models | Ibrahim, 2024 | [arXiv:2403.08763](https://arxiv.org/abs/2403.08763) |

## Part V. Supervised fine-tuning

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 13 | Finetuned Language Models Are Zero-Shot Learners (FLAN) | Wei, 2021 | [arXiv:2109.01652](https://arxiv.org/abs/2109.01652) |
| 13 | Multitask Prompted Training Enables Zero-Shot Task Generalization (T0) | Sanh, 2021 | [arXiv:2110.08207](https://arxiv.org/abs/2110.08207) |
| 13 | Scaling Instruction-Finetuned Language Models (Flan-T5, Flan-PaLM) | Chung, 2022 | [arXiv:2210.11416](https://arxiv.org/abs/2210.11416) |
| 13 | The Flan Collection: Designing Data and Methods for Effective Instruction Tuning | Longpre, 2023 | [arXiv:2301.13688](https://arxiv.org/abs/2301.13688) |
| 14 | Self-Instruct: Aligning Language Models with Self-Generated Instructions | Wang, 2022 | [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) |
| 14 | Alpaca: A Strong, Replicable Instruction-Following Model | Taori, 2023 | [Stanford CRFM blog](https://crfm.stanford.edu/2023/03/13/alpaca.html) |
| 14 | LIMA: Less Is More for Alignment | Zhou, 2023 | [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) |
| 15 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Lambert, 2024 | [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) |

## Part VI. Parameter-efficient tuning

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 16 | Parameter-Efficient Transfer Learning for NLP (adapters) | Houlsby, 2019 | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| 16 | Prefix-Tuning: Optimizing Continuous Prompts for Generation | Li, 2021 | [arXiv:2101.00190](https://arxiv.org/abs/2101.00190) |
| 16 | LoRA: Low-Rank Adaptation of Large Language Models | Hu, 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| 17 | QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers, 2023 | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| 17 | DoRA: Weight-Decomposed Low-Rank Adaptation | Liu, 2024 | [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) |
| 17 | LoRA Learns Less and Forgets Less | Biderman, 2024 | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) |

## Part VII. Reward models and RLHF

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 18 | Deep reinforcement learning from human preferences | Christiano, 2017 | [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) |
| 18 | Fine-Tuning Language Models from Human Preferences | Ziegler, 2019 | [arXiv:1909.08593](https://arxiv.org/abs/1909.08593) |
| 18 | Learning to summarize from human feedback | Stiennon, 2020 | [arXiv:2009.01325](https://arxiv.org/abs/2009.01325) |
| 19 | Proximal Policy Optimization Algorithms | Schulman, 2017 | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| 19 | High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE) | Schulman, 2015 | [arXiv:1506.02438](https://arxiv.org/abs/1506.02438) |
| 19 | Training language models to follow instructions with human feedback (InstructGPT) | Ouyang, 2022 | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| 20 | Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | Bai, 2022 | [arXiv:2204.05862](https://arxiv.org/abs/2204.05862) |
| 20 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron, 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| 20 | Scaling Laws for Reward Model Overoptimization | Gao, 2022 | [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) |
| 21 | Constitutional AI: Harmlessness from AI Feedback | Bai, 2022 | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| 21 | RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | Lee, 2023 | [arXiv:2309.00267](https://arxiv.org/abs/2309.00267) |

## Part VIII. Direct preference methods

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 22 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafailov, 2023 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| 23 | A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO) | Azar, 2023 | [arXiv:2310.12036](https://arxiv.org/abs/2310.12036) |
| 23 | KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh, 2024 | [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) |
| 23 | ORPO: Monolithic Preference Optimization without Reference Model | Hong, 2024 | [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) |
| 23 | SimPO: Simple Preference Optimization with a Reference-Free Reward | Meng, 2024 | [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) |
| 24 | Self-Rewarding Language Models | Yuan, 2024 | [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) |
| 24 | Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study | Xu, 2024 | [arXiv:2404.10719](https://arxiv.org/abs/2404.10719) |
| 24 | Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback | Ivison, 2024 | [arXiv:2406.09279](https://arxiv.org/abs/2406.09279) |

## Part IX. RL for reasoning

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 25 | Training Verifiers to Solve Math Word Problems (GSM8K) | Cobbe, 2021 | [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| 25 | STaR: Bootstrapping Reasoning With Reasoning | Zelikman, 2022 | [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) |
| 25 | Let's Verify Step by Step | Lightman, 2023 | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) |
| 25 | Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations | Wang, 2023 | [arXiv:2312.08935](https://arxiv.org/abs/2312.08935) |
| 26 | DeepSeekMath (GRPO), listed under Part II | Shao, 2024 | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| 26 | Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO) | Ahmadian, 2024 | [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) |
| 27 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI, 2025 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| 27 | Kimi k1.5: Scaling Reinforcement Learning with LLMs | Kimi Team, 2025 | [arXiv:2501.12599](https://arxiv.org/abs/2501.12599) |
| 27 | Tulu 3 (RLVR), listed under Part V | Lambert, 2024 | [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) |
| 28 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale | Yu, 2025 | [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) |
| 28 | Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO) | Liu, 2025 | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| 28 | Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters | Snell, 2024 | [arXiv:2408.03314](https://arxiv.org/abs/2408.03314) |

## Part X. Distillation

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 29 | Distilling the Knowledge in a Neural Network | Hinton, 2015 | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| 29 | Sequence-Level Knowledge Distillation | Kim, 2016 | [arXiv:1606.07947](https://arxiv.org/abs/1606.07947) |
| 29 | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter | Sanh, 2019 | [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) |
| 30 | MiniLLM: On-Policy Distillation of Large Language Models | Gu, 2023 | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| 30 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD) | Agarwal, 2023 | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) |
| 30 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team, 2024 | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| 30 | DeepSeek-R1 (distilled models), listed under Part IX | DeepSeek-AI, 2025 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |

## Part XI. Safety and evaluation

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 31 | Red Teaming Language Models with Language Models | Perez, 2022 | [arXiv:2202.03286](https://arxiv.org/abs/2202.03286) |
| 31 | Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions | Bianchi, 2023 | [arXiv:2309.07875](https://arxiv.org/abs/2309.07875) |
| 31 | Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! | Qi, 2023 | [arXiv:2310.03693](https://arxiv.org/abs/2310.03693) |
| 31 | Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG) | Zou, 2023 | [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) |
| 31 | Deliberative Alignment: Reasoning Enables Safer Language Models | Guan, 2024 | [arXiv:2412.16339](https://arxiv.org/abs/2412.16339) |
| 32 | Measuring Massive Multitask Language Understanding (MMLU) | Hendrycks, 2020 | [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| 32 | Evaluating Large Language Models Trained on Code (HumanEval, pass@k) | Chen, 2021 | [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) |
| 32 | Holistic Evaluation of Language Models (HELM) | Liang, 2022 | [arXiv:2211.09110](https://arxiv.org/abs/2211.09110) |
| 32 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Zheng, 2023 | [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) |
| 32 | Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference | Chiang, 2024 | [arXiv:2403.04132](https://arxiv.org/abs/2403.04132) |
| 32 | Proving Test Set Contamination in Black Box Language Models | Oren, 2023 | [arXiv:2310.17623](https://arxiv.org/abs/2310.17623) |

## Coverage and reward supplement — verified 2026-11-07

| Deck / lab | Paper | First author, year | Source / reading target |
|---|---|---|---|
| 26, 28 / L7 | Demystifying Reinforcement Learning Post-Training of Language Models | Clay, 2026 | [arXiv:2608.24949](https://arxiv.org/html/2608.24949v1) (v1), §§4.1–4.3 and 5.1–5.2 |

The paper motivates separating base-distribution, reward and prompt controls. Its controlled dense-reward results do not establish that arbitrary shaping improves general reasoning. L7's right-width proxy is a deliberately weak local counterexample, not the paper's edit-distance reward or PRM, and its arithmetic run is not a replication. DAPO §3.2 and Chen et al. §2.1 / Appendix A (already listed above) were also reread for zero-signal groups and the pass@k estimator.
