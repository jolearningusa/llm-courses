# Syllabus: LLM Transformer Architecture

Fifteen weeks, about three paper decks a week, with one lab session every week or two. The course also splits into two shorter courses: a **core** course (Parts 0 to V, weeks 1 to 8) and an **advanced** course (Parts VI to IX, weeks 9 to 15), which assumes the core.

Each paper deck follows five steps: the **problem** the authors faced, the architectural **change**, the **evidence** in the paper, the **modeling impact** (quality, parameters, FLOPs, KV memory, what it made possible), and the **afterlife** (which later models kept, changed or dropped it, checked against their configs). Papers are listed in the [bibliography](references/bibliography.md).

## Parts and decks

| Part | Deck | Paper(s) |
|---|---|---|
| 0 Map | 00 | Course map |
| I Foundations | 01 | Attention Is All You Need |
| | 02 | GPT-1 and GPT-2 |
| | 03 | BERT (with RoBERTa) |
| | 04 | T5 (with BART) |
| | 05 | GPT-3 and the Kaplan scaling laws |
| | 06 | Chinchilla |
| II Context | 07 | Subword tokenization: BPE, Unigram, SentencePiece |
| | 08 | RoPE (RoFormer) |
| | 09 | ALiBi |
| | 10 | Position Interpolation and YaRN |
| | 11 | From conversation to tokens: chat templates (InstructGPT onward), special tokens, packing, Lost in the Middle |
| III Attention | 12 | MQA |
| | 13 | GQA |
| | 14 | FlashAttention, as a modeling enabler |
| | 15 | Sliding windows and attention sinks: Mistral 7B, StreamingLLM |
| | 16 | MLA (DeepSeek-V2) |
| IV Block | 17 | Pre-LN and RMSNorm |
| | 18 | GLU variants (SwiGLU) |
| V MoE | 19 | Sparsely-gated MoE and GShard |
| | 20 | Switch Transformer |
| | 21 | Mixtral |
| | 22 | DeepSeekMoE and auxiliary-loss-free load balancing |
| VI Efficiency | 23 | Mamba and Mamba-2 |
| | 24 | Jamba |
| | 25 | Gated DeltaNet and Qwen3-Next |
| | 26 | Native Sparse Attention and DeepSeek Sparse Attention |
| | 27 | Multi-token prediction |
| VII Families | 28 | Llama 1 and 2 |
| | 29 | Llama 3 and 4 |
| | 30 | Qwen2.5 and Qwen3 |
| | 31 | DeepSeek-V3 and R1 |
| | 32 | Gemma 1, 2 and 3 |
| | 33 | Phi-3, OLMo, Kimi K2 and gpt-oss |
| VIII Multimodal | 34 | ViT |
| | 35 | CLIP |
| | 36 | Flamingo and LLaVA |
| | 37 | Chameleon |
| IX Beyond text | 38 | DDPM and latent diffusion |
| | 39 | DiT |
| | 40 | Whisper |
| | 41 | SoundStream and EnCodec |
| | 42 | AudioLM and VALL-E |
| | 43 | Capstone |

## Weekly plan

| Week | Decks | Lab |
|---|---|---|
| 1 | 00, 01 | L0: parameters and KV-cache bytes from a `config.json` |
| 2 | 02, 03, 04 | L1: one block in encoder, decoder and encoder-decoder modes |
| 3 | 05, 06, 07 | L2: byte-level BPE and token cost across languages |
| 4 | 08, 09, 10 | L3: sinusoidal, RoPE, ALiBi and Position Interpolation |
| 5 | 11, 12 | L4: chat templates and packing masks by hand |
| 6 | 13, 14, 15, 16 | L5: MHA, MQA, GQA and MLA with a KV cache |
| 7 | 17, 18 | L6: block variants on a character-level model |
| 8 | 19, 20, 21, 22 | L7: top-k MoE with a load-balancing loss |
| 9 | 23, 24 | Paper critique 1 |
| 10 | 25, 26, 27 | |
| 11 | 28, 29, 30 | Read the released configs side by side (extends L0) |
| 12 | 31, 32, 33 | Paper critique 2 |
| 13 | 34, 35, 36, 37 | L8: ViT patches and a CLIP-style contrastive loss |
| 14 | 38, 39, 40, 41, 42 | L9: residual vector quantization of audio |
| 15 | 43 | Capstone presentations |

## Labs

All labs run on a laptop CPU with NumPy or PyTorch and tiny models. Each lab checks its output against a reference implementation or a value computed by hand.

| Lab | Task | Check |
|---|---|---|
| L0 | Compute parameter count and KV-cache bytes per token from Hugging Face `config.json` files | Matches the parameter count of each released checkpoint |
| L1 | A tiny transformer from scratch that runs as encoder, decoder and encoder-decoder from one block | Causal mask leaks no future tokens; attention matches a one-query-at-a-time loop |
| L2 | Train a byte-level BPE tokenizer and measure tokens per character across languages | Round-trip decoding is lossless |
| L3 | Implement sinusoidal, RoPE and ALiBi; apply Position Interpolation and YaRN's per-frequency scaling | RoPE attention scores depend only on relative position; interpolated angles stay in the trained range |
| L4 | Render a chat template by hand and build packing masks | Equal to the published Llama 3 and Qwen2.5 formats; no attention across packed documents |
| L5 | MHA, MQA, GQA and MLA with a KV cache, plus window masks and blockwise attention | Cached decoding equals full recomputation; KV bytes match the formula; blockwise equals direct attention |
| L6 | Train small character-level models and compare post-LN, pre-LN, RMSNorm, SwiGLU, GQA and MQA | Fixed seeds and budget; report final loss with spread across seeds, and loss curves |
| L7 | A top-k MoE layer with capacity, shared experts, a load-balancing loss and loss-free bias balancing | Switch loss is 1 under uniform routing; both balancing methods lower MaxVio; Mixtral's 47B total and 13B active parameters reproduced |
| L8 | ViT patch embedding and a CLIP-style contrastive loss on toy data | Loss reaches the value expected for perfect pairing |
| L9 | Residual vector quantization of an audio signal | Reconstruction error falls as codebooks are added |

## Assessment

Labs 50%, two paper critiques 20%, capstone 30%.

- **Paper critique.** Pick a paper not covered in class and write two pages in the five-step format, including which earlier deck's component it changes and one claim you checked against its tables or released config.
- **Capstone (deck 43).** Design a model for a given parameter, KV-memory and modality budget. Defend each choice (layout, attention variant, positions, norm, FFN or MoE, tokenizer) with the papers in this course, and compute the parameter count and KV-cache size with the L0 tools.
