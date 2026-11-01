# LLM Transformer Architecture

Read the architecture papers behind today's language, vision, diffusion and speech models one at a time: why each change was proposed, what it changed in the model, and what later models kept.

**Status, 2026-11-01: Part I published.** The course map, the six Part I paper decks (the 2017 Transformer, GPT-1/2, BERT, T5, GPT-3 with the Kaplan scaling laws, Chinchilla), labs L0 and L1, the syllabus and the full bibliography (70 sources, each checked on its arXiv or vendor page) are in place. Parts II to IX follow part by part, in the order of the [syllabus](SYLLABUS.md). Decks are in English first; Simplified Chinese twins follow once the English decks are complete.

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

Engineers and students who use LLMs and want to read a model's technical report or `config.json` and understand every choice in it. You need Python, NumPy or PyTorch, linear algebra and basic probability. All labs run on a laptop CPU with tiny models.

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

Decks 07 to 43 are listed in the [syllabus](SYLLABUS.md) and are added to this table as each part is published.

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
│   └── 06_chinchilla.html
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
