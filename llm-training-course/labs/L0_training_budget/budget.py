"""A training budget from four numbers: parameters, tokens, GPUs and their speed.

    python budget.py --params 405e9 --tokens 15.6e12 --gpus 16384 --peak 989e12 --mfu 0.40
    python budget.py --compute 5.88e23                 # Chinchilla-optimal split of a FLOP budget
    python budget.py --params 7.5e9 --dp 64            # per-GPU model-state memory under ZeRO
    python budget.py --seq 2048 --hidden 12288 --heads 96 --tp 8   # activation memory per layer

Standard library only. Every formula names the deck and paper it comes from.
"""
from __future__ import annotations

import argparse
import math

GB = 1e9


# Deck 01, architecture deck 05: forward + backward costs about 6 FLOPs per parameter per token.
def train_flops(n_params: float, n_tokens: float) -> float:
    return 6.0 * n_params * n_tokens


# Architecture deck 06 (Chinchilla): about 20 tokens per parameter at the compute optimum.
# With C = 6ND and D = 20N, C = 120 N^2.
def chinchilla_optimal(compute: float, tokens_per_param: float = 20.0) -> tuple[float, float]:
    n = math.sqrt(compute / (6.0 * tokens_per_param))
    return n, tokens_per_param * n


def train_days(compute: float, gpus: int, peak_flops: float, mfu: float) -> float:
    """Wall-clock days at a model FLOPs utilization (MFU, PaLM Appendix B)."""
    return compute / (gpus * peak_flops * mfu) / 86400.0


def mfu(tokens_per_second: float, n_params: float, gpus: int, peak_flops: float) -> float:
    """Model FLOPs utilization from measured throughput, ignoring the attention term."""
    return 6.0 * n_params * tokens_per_second / (gpus * peak_flops)


# Deck 09, ZeRO (arXiv:1910.02054) section 3.1 and Figure 1: mixed-precision Adam keeps
# 2 bytes of fp16 weights, 2 of fp16 gradients and K = 12 of fp32 master weights, momentum, variance.
def model_state_bytes(n_params: float, stage: int, dp: int, k: int = 12) -> float:
    """Per-GPU bytes for weights, gradients and optimizer state at ZeRO stage 0 to 3."""
    psi = n_params
    if stage == 0:
        return (2 + 2 + k) * psi
    if stage == 1:
        return 2 * psi + 2 * psi + k * psi / dp
    if stage == 2:
        return 2 * psi + (2 + k) * psi / dp
    if stage == 3:
        return (2 + 2 + k) * psi / dp
    raise ValueError("stage must be 0, 1, 2 or 3")


# Deck 09, GPipe (arXiv:1811.06965) section 2.3: with K pipeline stages and M micro-batches the
# idle "bubble" is (K-1)/(M+K-1) of the step. Narayanan et al. (arXiv:2104.04473) section 2.2 write
# it as bubble time over ideal time, (p-1)/m, and divide it by v for the interleaved schedule.
def pipeline_bubble(stages: int, micro_batches: int) -> float:
    """Fraction of a GPipe step that each stage spends idle."""
    return (stages - 1) / (micro_batches + stages - 1)


def bubble_over_ideal(stages: int, micro_batches: int, chunks: int = 1) -> float:
    """Narayanan et al.: bubble time divided by ideal compute time, with v chunks per device."""
    return (stages - 1) / (micro_batches * chunks)


# Deck 09, Narayanan et al. (arXiv:2104.04473) eq. 4: with activation recomputation a step costs
# about 8 FLOPs per parameter per token (6 for forward and backward, 2 to redo the forward).
def recompute_days(n_params: float, n_tokens: float, gpus: int, flops_per_gpu: float) -> float:
    """End-to-end days at a measured per-GPU throughput, 8TP / (nX)."""
    return 8.0 * n_tokens * n_params / (gpus * flops_per_gpu) / 86400.0


# Deck 09, Korthikanti et al. (arXiv:2205.05198) eqs. 1, 2 and 4 and Table 2: activation bytes per
# Transformer layer in 16-bit, for sequence s, micro-batch b, hidden h, heads a, tensor-parallel t.
def activation_bytes(s: int, b: int, h: int, a: int, t: int = 1,
                     sequence_parallel: bool = False, selective: bool = False) -> float:
    attn = 0.0 if selective else 5.0 * a * s / h   # selective recomputation drops the s^2 term
    if sequence_parallel:
        return s * b * h * (34.0 + attn) / t
    return s * b * h * (10.0 + 24.0 / t + attn / t)


# Deck 02, McCandlish et al. (arXiv:1812.06162) eq. 2.11: (S/Smin - 1)(E/Emin - 1) = 1,
# with B_crit = Emin/Smin. Returns the steps and examples a batch size B costs.
def steps_and_examples(batch: float, s_min: float, e_min: float) -> tuple[float, float]:
    b_crit = e_min / s_min
    return s_min * (1 + b_crit / batch), e_min * (1 + batch / b_crit)


# Deck 04, Muon (Jordan 2024): Newton-Schulz costs at most T*m/B of the forward-backward FLOPs,
# for T iterations, model width m and batch size B in tokens.
def muon_overhead(width: int, batch_tokens: int, ns_steps: int = 5) -> float:
    return ns_steps * width / batch_tokens


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--params", type=float)
    p.add_argument("--tokens", type=float)
    p.add_argument("--compute", type=float, help="FLOP budget to split Chinchilla-optimally")
    p.add_argument("--gpus", type=int)
    p.add_argument("--peak", type=float, help="peak FLOP/s per GPU at the training precision")
    p.add_argument("--mfu", type=float, default=0.4)
    p.add_argument("--dp", type=int, help="data-parallel degree for the ZeRO table")
    p.add_argument("--seq", type=int, help="sequence length for the activation table")
    p.add_argument("--hidden", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--micro", type=int, default=1, help="micro-batch size in sequences")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    a = p.parse_args()

    if a.compute:
        n, d = chinchilla_optimal(a.compute)
        print(f"Chinchilla-optimal for {a.compute:.3g} FLOPs: {n / 1e9:.1f}B parameters, {d / 1e12:.2f}T tokens")
    if a.params and a.tokens:
        c = train_flops(a.params, a.tokens)
        print(f"6ND = {c:.3g} FLOPs, {a.tokens / a.params:.0f} tokens per parameter")
        if a.gpus and a.peak:
            print(f"{train_days(c, a.gpus, a.peak, a.mfu):.1f} days on {a.gpus} GPUs at {a.mfu:.0%} MFU")
    if a.params and a.dp:
        for stage in range(4):
            print(f"ZeRO stage {stage}: {model_state_bytes(a.params, stage, a.dp) / GB:8.2f} GB per GPU")
    if a.seq and a.hidden and a.heads:
        args = (a.seq, a.micro, a.hidden, a.heads, a.tp)
        for name, sp, sel in [("tensor parallel", False, False), ("+ sequence parallel", True, False),
                              ("+ selective recompute", True, True)]:
            got = activation_bytes(*args, sequence_parallel=sp, selective=sel)
            print(f"{name:22s} {got / GB:7.2f} GB per layer per GPU")


if __name__ == "__main__":
    main()
