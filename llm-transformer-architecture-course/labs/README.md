# Labs

CPU-only labs, each checked against a reference value or a definition. They are published with the part
of the course they belong to; the [syllabus](../SYLLABUS.md) lists all ten.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_config_math/) | Parameters, KV-cache bytes and training compute from a `config.json` | 01, 05, 06 | Python 3.10+ |
| [L1](L1_one_block/) | One block run as encoder, decoder and encoder-decoder | 01–04 | Python 3.10+, NumPy |
| [L2](L2_bpe/) | Byte-level BPE and token cost across languages | 07 | Python 3.10+ |
| [L3](L3_positions/) | Sinusoidal, RoPE, ALiBi, Position Interpolation and YaRN | 08–10 | Python 3.10+, NumPy |
| [L4](L4_chat_templates/) | Chat templates, packing, document and loss masks | 11 | Python 3.10+, NumPy |
| [L5](L5_attention_variants/) | MHA, MQA, GQA and MLA with a KV cache; windows, sinks and blockwise attention | 12–16 | Python 3.10+, NumPy |
| [L6](L6_block_variants/) | Train small models with Post-LN, Pre-LN, RMSNorm, SwiGLU, GQA and MQA | 12, 13, 17, 18 | Python 3.10+, PyTorch 2 (CPU) |
| [L7](L7_moe/) | Top-k MoE with capacity, shared experts, the Switch loss and loss-free balancing | 19–22 | Python 3.10+, NumPy |

Run every lab's checks from the lab directory with `python -m unittest`.
