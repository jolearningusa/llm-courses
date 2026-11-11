# Bibliography

Every paper the course reads, grouped by the part of the course that reads it. Original entries were checked
on 2026-11-01 (arXiv ID, title and first author, or the blog's own page); dated supplements record their own checks. The year is that of arXiv v1, or of publication for sources without an arXiv paper. Decks cite
sections and tables of these papers; model facts such as layer counts come from the paper's tables or the
released `config.json`.

## Part I. Foundations and the three families

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 01 | Attention Is All You Need | Vaswani, 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 02 | Improving Language Understanding by Generative Pre-Training (GPT-1) | Radford, 2018 | [OpenAI PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 02 | Language Models are Unsupervised Multitask Learners (GPT-2) | Radford, 2019 | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| 03 | BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin, 2018 | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |
| 03 | RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu, 2019 | [arXiv:1907.11692](https://arxiv.org/abs/1907.11692) |
| 04 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) | Raffel, 2019 | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| 04 | BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension | Lewis, 2019 | [arXiv:1910.13461](https://arxiv.org/abs/1910.13461) |
| 05 | Language Models are Few-Shot Learners (GPT-3) | Brown, 2020 | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| 05 | Scaling Laws for Neural Language Models | Kaplan, 2020 | [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) |
| 06 | Training Compute-Optimal Large Language Models (Chinchilla) | Hoffmann, 2022 | [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) |

## Part II. How the context is constructed

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 07 | Neural Machine Translation of Rare Words with Subword Units (BPE) | Sennrich, 2015 | [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) |
| 07 | Subword Regularization (Unigram language model tokenizer) | Kudo, 2018 | [arXiv:1804.10959](https://arxiv.org/abs/1804.10959) |
| 07 | SentencePiece: A simple and language independent subword tokenizer and detokenizer | Kudo, 2018 | [arXiv:1808.06226](https://arxiv.org/abs/1808.06226) |
| 08 | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su, 2021 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 09 | Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi) | Press, 2021 | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | Extending Context Window of Large Language Models via Positional Interpolation | Chen, 2023 | [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) |
| 10 | YaRN: Efficient Context Window Extension of Large Language Models | Peng, 2023 | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 11 | Training language models to follow instructions with human feedback (InstructGPT) | Ouyang, 2022 | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| 11 | Lost in the Middle: How Language Models Use Long Contexts | Liu, 2023 | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) |

## Part III. Attention variants and the KV cache

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 12 | Fast Transformer Decoding: One Write-Head is All You Need (MQA) | Shazeer, 2019 | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 13 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie, 2023 | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 14 | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | Dao, 2022 | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 15 | Mistral 7B | Jiang, 2023 | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| 15 | Efficient Streaming Language Models with Attention Sinks (StreamingLLM) | Xiao, 2023 | [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) |
| 16 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA) | DeepSeek-AI, 2024 | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |

## Part IV. The modern block

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 17 | On Layer Normalization in the Transformer Architecture (Pre-LN) | Xiong, 2020 | [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) |
| 17 | Root Mean Square Layer Normalization (RMSNorm) | Zhang, 2019 | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) |
| 18 | GLU Variants Improve Transformer (SwiGLU) | Shazeer, 2020 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) |

## Part V. Mixture of experts

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 19 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | Shazeer, 2017 | [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) |
| 19 | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding | Lepikhin, 2020 | [arXiv:2006.16668](https://arxiv.org/abs/2006.16668) |
| 20 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity | Fedus, 2021 | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) |
| 21 | Mixtral of Experts | Jiang, 2024 | [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) |
| 22 | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | Dai, 2024 | [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) |
| 22 | Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts | Wang, 2024 | [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) |

## Part VI. Newer efficiency architectures

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 23 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu, 2023 | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 23 | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2) | Dao, 2024 | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| 24 | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber, 2024 | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| 25 | Gated Delta Networks: Improving Mamba2 with Delta Rule | Yang, 2024 | [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 25 | Qwen3-Next: Towards Ultimate Training & Inference Efficiency | Qwen team, 2025 | [Alibaba Cloud blog](https://www.alibabacloud.com/blog/602580) |
| 26 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | Yuan, 2025 | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) |
| 26 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek Sparse Attention) | DeepSeek-AI, 2025 | [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) |
| 27 | Better & Faster Large Language Models via Multi-token Prediction | Gloeckle, 2024 | [arXiv:2404.19737](https://arxiv.org/abs/2404.19737) |

## Part VII. Popular model families

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 28 | LLaMA: Open and Efficient Foundation Language Models | Touvron, 2023 | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |
| 28 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron, 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| 29 | The Llama 3 Herd of Models | Grattafiori, 2024 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| 29 | The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation | Meta, 2025 | [Meta AI blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| 30 | Qwen2.5 Technical Report | Qwen team, 2024 | [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) |
| 30 | Qwen3 Technical Report | Yang, 2025 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| 31 | DeepSeek-V3 Technical Report | DeepSeek-AI, 2024 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| 31 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI, 2025 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| 32 | Gemma: Open Models Based on Gemini Research and Technology | Gemma Team, 2024 | [arXiv:2403.08295](https://arxiv.org/abs/2403.08295) |
| 32 | Gemma 2: Improving Open Language Models at a Practical Size | Gemma Team, 2024 | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| 32 | Gemma 3 Technical Report | Gemma Team, 2025 | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| 33 | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone | Abdin, 2024 | [arXiv:2404.14219](https://arxiv.org/abs/2404.14219) |
| 33 | OLMo: Accelerating the Science of Language Models | Groeneveld, 2024 | [arXiv:2402.00838](https://arxiv.org/abs/2402.00838) |
| 33 | Kimi K2: Open Agentic Intelligence | Kimi Team (Bai), 2025 | [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) |
| 33 | gpt-oss-120b & gpt-oss-20b Model Card | OpenAI, 2025 | [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) |

## Part VIII. Multimodal models

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 34 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT) | Dosovitskiy, 2020 | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) |
| 35 | Learning Transferable Visual Models From Natural Language Supervision (CLIP) | Radford, 2021 | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| 36 | Flamingo: a Visual Language Model for Few-Shot Learning | Alayrac, 2022 | [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| 36 | Visual Instruction Tuning (LLaVA) | Liu, 2023 | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) |
| 37 | Chameleon: Mixed-Modal Early-Fusion Foundation Models | Chameleon Team, 2024 | [arXiv:2405.09818](https://arxiv.org/abs/2405.09818) |

## Part IX. Transformers beyond text

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 38 | Denoising Diffusion Probabilistic Models (DDPM) | Ho, 2020 | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) |
| 38 | High-Resolution Image Synthesis with Latent Diffusion Models | Rombach, 2021 | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) |
| 39 | Scalable Diffusion Models with Transformers (DiT) | Peebles, 2022 | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) |
| 40 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | Radford, 2022 | [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) |
| 41 | SoundStream: An End-to-End Neural Audio Codec | Zeghidour, 2021 | [arXiv:2107.03312](https://arxiv.org/abs/2107.03312) |
| 41 | High Fidelity Neural Audio Compression (EnCodec) | Défossez, 2022 | [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) |
| 42 | AudioLM: a Language Modeling Approach to Audio Generation | Borsos, 2022 | [arXiv:2209.03143](https://arxiv.org/abs/2209.03143) |
| 42 | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E) | Wang, 2023 | [arXiv:2301.02111](https://arxiv.org/abs/2301.02111) |

## Related courses in this repository

- Attention kernels, KV-cache paging, MoE execution and serving: [Efficient LLM Computing Systems](../../llm-compute-systems-course/).
- How agents fill and manage the context window: [LLM AgentOS](../../llm-agentos-course/).

## Long-context evaluation supplement — checked 2026-11-07

| Deck | Paper | First author, year | Source |
|---|---|---|---|
| 08–10, 15 | RULER: What’s the Real Context Size of Your Long-Context Language Models? | Hsieh, 2024 | [arXiv:2404.06654](https://arxiv.org/html/2404.06654v1) |

Reading scope: v1 task families, effective-length methodology and error analysis; [author code](https://github.com/NVIDIA/RULER). The course fixture grid is original and much smaller, not RULER or a reproduction of its reported model results. Historical results are not current model rankings.


## Optional Mamba-3 supplement — verified 2026-11-08

| Deck / lab | Paper | First author, year | Source / reading target |
|---|---|---|---|
| 23 / L6 optional algebra | Mamba-3: Improved Sequence Modeling using State Space Principles | Lahoti, 2026 | [arXiv:2603.15569](https://arxiv.org/html/2603.15569v1) (v1), §§3.1–3.3, Appendix C; §§4 / G for separately scoped experiments |

Additional code reading: [author Mamba3 module at e9594ce](https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/modules/mamba3.py), `step` and `allocate_inference_cache`. Its core SSM cache omits the rank dimension while the auxiliary K cache includes it; phase and V caches are also present. The step comment limits its tested hardware to H100 at this revision. Source inspection is not a local compatibility or speed test.

The new diagrams are original teaching schematics. No new paper result table or speed ratio is reproduced. The local scalar/complex recurrence and separate MIMO matrix fragments are mathematical fixtures, not a full learned architecture or the author's kernel.

## Byte/chunk supplement — verified 2026-11-10

| Deck / lab | Paper | First author, year | Reading target |
|---|---|---|---|
| 07 / L2 optional | Byte Latent Transformer: Patches Scale Better Than Tokens | Pagnoni, 2024 | [arXiv:2412.09871](https://arxiv.org/html/2412.09871v1) (v1), §§2–4, Fig. 3/5 |
| 07 / L2 optional | Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | Hwang, 2025 | [arXiv:2507.07955](https://arxiv.org/html/2507.07955v1) (v1), §§2.2/2.4, Fig. 1/2 |

The [dated source/execution note](byte_chunk_accounting_2026_11.md) pins inspected author files, records their hashes, attributes the original inline diagrams, and separates actual CPU segmentation/algebra from declared neural-stage counts and unmeasured model quality/performance. No author implementation was executed.
