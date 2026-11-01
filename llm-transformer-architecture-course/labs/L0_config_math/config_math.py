"""Lab L0: read a model's config.json and compute what it implies.

    python config_math.py configs/*.json            # table for every config
    python config_math.py configs/llama-3-8b.json --context 8192 --tokens 15e12

Everything here is arithmetic on the config. Nothing is downloaded and no
weights are loaded. Supported layouts: GPT-2 (learned positions, LayerNorm,
GELU MLP, biases, tied embeddings) and the Llama family (Llama, Mistral, Qwen2:
RoPE, RMSNorm, SwiGLU MLP, grouped-query attention).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Arch:
    name: str
    layers: int
    d_model: int
    heads: int
    kv_heads: int
    d_ff: int
    vocab: int
    max_positions: int
    gated_mlp: bool          # SwiGLU (3 matrices) instead of a 2-matrix MLP
    qkv_bias: bool           # biases on the q, k, v projections
    all_bias: bool           # biases on every linear layer (GPT-2)
    layernorm: bool          # LayerNorm (weight + bias) instead of RMSNorm (weight only)
    learned_positions: bool  # a learned position-embedding table
    tied_embeddings: bool    # output head shares the input embedding matrix

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads


def load(path: str | Path) -> Arch:
    cfg = json.loads(Path(path).read_text())
    kind = cfg["model_type"]
    if kind == "gpt2":
        d = cfg["n_embd"]
        return Arch(Path(path).stem, cfg["n_layer"], d, cfg["n_head"], cfg["n_head"],
                    cfg.get("n_inner") or 4 * d, cfg["vocab_size"], cfg["n_positions"],
                    gated_mlp=False, qkv_bias=True, all_bias=True, layernorm=True,
                    learned_positions=True, tied_embeddings=True)
    if kind in ("llama", "mistral", "qwen2"):
        return Arch(Path(path).stem, cfg["num_hidden_layers"], cfg["hidden_size"],
                    cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
                    cfg["intermediate_size"], cfg["vocab_size"], cfg["max_position_embeddings"],
                    gated_mlp=True,
                    # Qwen2 always has q/k/v biases; Llama has them only when attention_bias is set.
                    qkv_bias=(kind == "qwen2") or bool(cfg.get("attention_bias", False)),
                    all_bias=False, layernorm=False, learned_positions=False,
                    tied_embeddings=bool(cfg.get("tie_word_embeddings", False)))
    raise ValueError(f"{path}: model_type {kind!r} is not supported")


def params(a: Arch) -> dict[str, int]:
    """Parameter count by component. Must match the released model exactly."""
    d, hd = a.d_model, a.head_dim
    kv = a.kv_heads * hd                        # width of K (and of V) after projection
    attn = d * d + 2 * d * kv + d * d           # W_q, W_k, W_v, W_o
    if a.all_bias:
        attn += d + 2 * kv + d
    elif a.qkv_bias:
        attn += d + 2 * kv
    if a.gated_mlp:
        mlp = 3 * d * a.d_ff                    # gate, up, down
    else:
        mlp = 2 * d * a.d_ff + (a.d_ff + d if a.all_bias else 0)
    norm = 2 * d if a.layernorm else d          # per norm layer
    block = attn + mlp + 2 * norm
    embed = a.vocab * d
    out = {
        "embedding": embed,
        "positions": a.max_positions * d if a.learned_positions else 0,
        "blocks": a.layers * block,
        "final_norm": norm,
        "lm_head": 0 if a.tied_embeddings else a.vocab * d,
    }
    out["total"] = sum(out.values())
    out["non_embedding"] = out["blocks"] + out["final_norm"]  # Kaplan's N
    return out


def kv_bytes_per_token(a: Arch, bytes_per_value: int = 2) -> int:
    """K and V for one token across all layers: 2 x layers x kv_heads x head_dim."""
    return 2 * a.layers * a.kv_heads * a.head_dim * bytes_per_value


def train_flops(n_params: int, tokens: float) -> float:
    """Kaplan's approximation C = 6 N D (forward 2N, backward 4N per token)."""
    return 6 * n_params * tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--context", type=int, default=None, help="context length for the KV-cache column (default: the config's maximum)")
    ap.add_argument("--tokens", type=float, default=None, help="training tokens, for a 6ND compute estimate")
    args = ap.parse_args(argv)

    print(f"{'model':18} {'params':>15} {'non-emb':>15} {'GQA':>6} {'KV/token':>10} {'KV@ctx':>10} {'ctx':>7}")
    for path in args.configs:
        a = load(path)
        p = params(a)
        ctx = args.context or a.max_positions
        kv = kv_bytes_per_token(a)
        print(f"{a.name:18} {p['total']:>15,} {p['non_embedding']:>15,} {a.heads // a.kv_heads:>5}x "
              f"{kv / 2**10:>8.0f}KiB {kv * ctx / 2**30:>8.2f}GiB {ctx:>7}")
        if args.tokens:
            print(f"{'':18} training compute 6ND = {train_flops(p['total'], args.tokens):.2e} FLOPs, "
                  f"{args.tokens / p['total']:.0f} tokens per parameter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
