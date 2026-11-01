# LLM Transformer Architecture

Read the architecture papers behind today's language, vision, diffusion and speech models one at a time: why each change was proposed, what it changed in the model, and what later models kept.

**Status, 2026-11-01: Parts I to VI published.** The course map, the six Part I paper decks (the 2017 Transformer, GPT-1/2, BERT, T5, GPT-3 with the Kaplan scaling laws, Chinchilla), the five Part II decks on how the context is constructed (subword tokenization, RoPE, ALiBi, Position Interpolation and YaRN, chat templates and long contexts), the seven Part III and IV decks on attention variants and the modern block (MQA, GQA, FlashAttention, sliding windows and attention sinks, MLA, Pre-LN and RMSNorm, SwiGLU), the four Part V decks on mixture of experts (sparsely-gated MoE and GShard, Switch, Mixtral, DeepSeekMoE with loss-free balancing), the five Part VI decks on newer efficiency designs (Mamba and Mamba-2, Jamba, Gated DeltaNet and Qwen3-Next, native and DeepSeek sparse attention, multi-token prediction), labs L0 to L7, the syllabus and the full bibliography (70 sources, each checked on its arXiv or vendor page) are in place. Parts VII to IX follow part by part, in the order of the [syllabus](SYLLABUS.md). Decks are in English first; Simplified Chinese twins follow once the English decks are complete.

## Start here

- [Course map deck](slides/00_course_map.html): the whole course on 11 slides.
- [Syllabus](SYLLABUS.md): parts, weekly plan, labs and assessment.
- [Bibliography](references/bibliography.md): every paper, grouped by the deck that reads it.
- [Labs](labs/): CPU-only exercises with their own checks (`python -m unittest` in each lab folder).

## What the course covers

The course takes one model apart from the input text to the next token and then follows each component through the papers that shaped it.

- **Foundations.** The 2017 Transformer and the three families it produced: encoder-only (BERT), decoder-only (GPT) and encoder-decoder (T5), then the scaling laws that pushed models toward large decoders.
- **How the context is constructed.** Subword tokenization, rotary and linear-bias position encodings, context-window extension, chat templates and what models do with long inputs.
- **Attention variants and the KV cache.** MQA, GQA, IO-aware exact attention, sliding windows and attention sinks, and multi-head latent attention (MLA).
- **The modern block and MoE.** Pre-norm and RMSNorm, gated FFNs, and the mixture-of-experts line from the sparsely-gated layer to DeepSeekMoE.
- **Newer efficiency designs.** State-space models, hybrids, gated linear attention, trainable sparse attention and multi-token prediction.
- **Model families, report by report.** Llama, Qwen, DeepSeek, Gemma, and Phi, OLMo, Kimi K2 and gpt-oss.
- **Beyond text.** ViT, CLIP, vision-language models, diffusion transformers, Whisper, neural audio codecs and codec language models.

Every paper deck follows the same five steps: the problem, the architectural change, the evidence in the paper, the modeling impact, and what later models kept or dropped.

## Audience and prerequisites

Engineers and students who use LLMs and want to read a model's technical report or `config.json` and understand every choice in it. You need Python, NumPy, linear algebra and basic probability; lab L6 also uses PyTorch. All labs run on a laptop CPU with tiny models.

Kernels, KV-cache paging, MoE execution and serving are taught in [Efficient LLM Computing Systems](../llm-compute-systems-course/). How agents fill and manage the context window is taught in [LLM AgentOS](../llm-agentos-course/). This course links to both instead of repeating them.

## Slide decks

| Deck | Teaching use |
|---|---|
| [Course map](slides/00_course_map.html) (11 slides) | Week 1: the pipeline from text to logits, what changed since 2017, how each paper deck is built |
| **Part I. Foundations and the three families** | |
| [01 Attention Is All You Need](slides/01_attention_is_all_you_need.html) (11 slides) | Week 1: attention, heads, positions, the encoder-decoder; lab L1 |
| [02 GPT-1 and GPT-2](slides/02_gpt1_gpt2.html) (10 slides) | Week 2: decoder-only pre-training, tasks as text, pre-LN |
| [03 BERT](slides/03_bert.html) (10 slides) | Week 2: encoder-only, masked LM, RoBERTa |
| [04 T5](slides/04_t5.html) (10 slides) | Week 2: encoder-decoder, text-to-text, span corruption, BART |
| [05 GPT-3 and the Kaplan scaling laws](slides/05_gpt3_scaling_laws.html) (10 slides) | Week 3: power laws, in-context learning; lab L0 |
| [06 Chinchilla](slides/06_chinchilla.html) (9 slides) | Week 3: compute-optimal parameters and tokens; lab L0 |
| **Part II. How the context is constructed** | |
| [07 Subword tokenization](slides/07_subword_tokenization.html) (9 slides) | Week 3: BPE, byte-level BPE, Unigram and SentencePiece, vocabulary size; lab L2 |
| [08 RoPE](slides/08_rope.html) (9 slides) | Week 4: rotary position embedding, relative position from rotation; lab L3 |
| [09 ALiBi](slides/09_alibi.html) (9 slides) | Week 4: a linear distance bias instead of position embeddings; lab L3 |
| [10 Position Interpolation and YaRN](slides/10_pi_yarn.html) (9 slides) | Week 4: extending a RoPE model's window after pre-training; lab L3 |
| [11 From conversation to tokens](slides/11_conversation_to_tokens.html) (10 slides) | Week 5: chat templates, packing masks, context windows, lost in the middle; lab L4 |
| **Part III. Attention variants and the KV cache** | |
| [12 Multi-query attention](slides/12_mqa.html) (9 slides) | Week 5: memory-bound decoding, one shared key and value head; lab L5 |
| [13 Grouped-query attention](slides/13_gqa.html) (9 slides) | Week 6: groups of query heads, uptraining from multi-head checkpoints; lab L5 |
| [14 FlashAttention](slides/14_flashattention.html) (9 slides) | Week 6: exact tiled attention and what it made trainable; lab L5 |
| [15 Sliding windows and attention sinks](slides/15_sliding_window_sinks.html) (10 slides) | Week 6: Mistral 7B's window, StreamingLLM's sinks; lab L5 |
| [16 Multi-head latent attention](slides/16_mla.html) (10 slides) | Week 6: a latent KV cache, absorbed projections, decoupled RoPE; lab L5 |
| **Part IV. The modern block** | |
| [17 Pre-LN and RMSNorm](slides/17_preln_rmsnorm.html) (10 slides) | Week 7: where the norm goes, why warm-up was needed, dropping the mean; lab L6 |
| [18 GLU variants and SwiGLU](slides/18_glu_variants.html) (9 slides) | Week 7: gated FFNs at equal parameters; lab L6 |
| **Part V. Mixture of experts** | |
| [19 Sparsely-gated MoE and GShard](slides/19_sparse_moe_gshard.html) (10 slides) | Week 8: routers, top-k gating, balance losses, MoE in the Transformer; lab L7 |
| [20 Switch Transformer](slides/20_switch.html) (10 slides) | Week 8: top-1 routing, expert capacity, stable sparse training; lab L7 |
| [21 Mixtral](slides/21_mixtral.html) (9 slides) | Week 8: a decoder-only MoE, total and active parameters, what the router learns; lab L7 |
| [22 DeepSeekMoE and loss-free balancing](slides/22_deepseekmoe.html) (10 slides) | Week 8: fine-grained and shared experts, bias-based balancing; lab L7 |
| **Part VI. Newer efficiency architectures** | |
| [23 Mamba and Mamba-2](slides/23_mamba.html) (10 slides) | Week 9: selective state spaces, the hardware-aware scan, state space duality; paper critique 1 |
| [24 Jamba](slides/24_jamba.html) (9 slides) | Week 9: a 1:7 attention-Mamba hybrid with MoE, KV cache at 256K |
| [25 Gated DeltaNet and Qwen3-Next](slides/25_gated_deltanet.html) (9 slides) | Week 10: the gated delta rule, a 3:1 linear-attention hybrid in production |
| [26 Native Sparse Attention and DeepSeek Sparse Attention](slides/26_sparse_attention.html) (10 slides) | Week 10: trainable sparse attention, block selection and a lightning indexer |
| [27 Multi-token prediction](slides/27_multi_token_prediction.html) (9 slides) | Week 10: predicting several tokens ahead, self-speculative decoding |

Decks 28 to 43 are listed in the [syllabus](SYLLABUS.md) and are added to this table as each part is published.

## Repository layout

```text
llm-transformer-architecture-course/
├── labs/
│   ├── L0_config_math/
│   │   ├── configs/
│   │   │   ├── gpt2.json
│   │   │   ├── llama-2-7b.json
│   │   │   ├── llama-3-8b.json
│   │   │   ├── mistral-7b-v0.1.json
│   │   │   └── qwen2.5-7b.json
│   │   ├── README.md
│   │   ├── config_math.py           # Parameters, KV-cache bytes and 6ND compute from a config
│   │   └── test_config_math.py
│   ├── L1_one_block/
│   │   ├── README.md
│   │   ├── test_transformer.py
│   │   └── transformer.py           # One NumPy block as encoder, decoder and encoder-decoder
│   ├── L2_bpe/
│   │   ├── corpus/                  # One passage in four languages
│   │   │   ├── de.txt
│   │   │   ├── en.txt
│   │   │   ├── es.txt
│   │   │   └── zh.txt
│   │   ├── README.md
│   │   ├── bpe.py                   # Byte-level BPE: train, encode, decode
│   │   └── test_bpe.py
│   ├── L3_positions/
│   │   ├── README.md
│   │   ├── positions.py             # Sinusoidal, RoPE, ALiBi, Position Interpolation, YaRN
│   │   └── test_positions.py
│   ├── L4_chat_templates/
│   │   ├── README.md
│   │   ├── chat.py                  # Llama 3 and Qwen2.5 templates, loss and packing masks
│   │   └── test_chat.py
│   ├── L5_attention_variants/
│   │   ├── README.md
│   │   ├── attention.py             # MHA, GQA, MQA, MLA with a KV cache; windows; blockwise attention
│   │   └── test_attention.py
│   ├── L6_block_variants/
│   │   ├── README.md
│   │   ├── model.py                 # Small PyTorch LM with switchable norm, placement, FFN and KV heads
│   │   ├── test_model.py
│   │   └── train.py                 # Trains the variants on the same data, budget and seeds
│   ├── L7_moe/
│   │   ├── README.md
│   │   ├── moe.py                   # Top-k MoE, capacity, shared experts, Switch loss, loss-free bias
│   │   └── test_moe.py
│   └── README.md                    # Lab index
├── references/
│   └── bibliography.md              # Every paper, grouped by deck; sources checked 2026-11-01
├── slides/
│   ├── 00_course_map.html           # Course map deck
│   ├── 01_attention_is_all_you_need.html
│   ├── 02_gpt1_gpt2.html
│   ├── 03_bert.html
│   ├── 04_t5.html
│   ├── 05_gpt3_scaling_laws.html
│   ├── 06_chinchilla.html
│   ├── 07_subword_tokenization.html
│   ├── 08_rope.html
│   ├── 09_alibi.html
│   ├── 10_pi_yarn.html
│   ├── 11_conversation_to_tokens.html
│   ├── 12_mqa.html
│   ├── 13_gqa.html
│   ├── 14_flashattention.html
│   ├── 15_sliding_window_sinks.html
│   ├── 16_mla.html
│   ├── 17_preln_rmsnorm.html
│   ├── 18_glu_variants.html
│   ├── 19_sparse_moe_gshard.html
│   ├── 20_switch.html
│   ├── 21_mixtral.html
│   ├── 22_deepseekmoe.html
│   ├── 23_mamba.html
│   ├── 24_jamba.html
│   ├── 25_gated_deltanet.html
│   ├── 26_sparse_attention.html
│   └── 27_multi_token_prediction.html
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
