# Lab L4: LoRA from scratch and 4-bit NormalFloat

Used with decks 16 and 17. CPU only, Python 3.10+ and NumPy.

LoRA trains two thin matrices beside a frozen weight; QLoRA stores that frozen weight in 4 bits. This lab
writes both in a few lines of NumPy: a LoRA layer with hand-written gradients, its merge into the base
weight, the NF4 code book built from normal quantiles, block-wise quantization, and double quantization
of the block constants.

| File | What it holds | Deck |
|---|---|---|
| `lora.py` | A LoRA layer (Gaussian A, zero B, scale α/r), its gradients, merging, the r(d + k) parameter count, and a fit against the best rank-r error | 16, 17 |
| `nf4.py` | The NF4 and Int4 code books, block-wise absmax quantization (blocks of 64), double quantization of the constants (8 bits, blocks of 256) | 17 |

## Run

```bash
python -m unittest test_lora_nf4.py    # about a second
python lora.py                         # fit ranks 1, 2, 4, 8 to an update of rank 4
python lora.py --true-rank 8 --ranks 1 2 4 8 16
```

## Checks

- With B = 0 at the start, the LoRA layer computes exactly the base model's output.
- The merged weight W0 + (α/r)BA gives the same outputs as the two-path layer.
- The gradients for A and B match finite differences.
- r(d + k) per matrix reproduces the 4.7M and 37.7M trainable parameters of LoRA on GPT-3's Wq and Wv at
  r = 1 and r = 8 ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685) Table 4).
- Below the true rank, the trained loss reaches the Eckart-Young bound (the energy in the discarded singular
  values) within 1%; at the true rank it reaches zero.
- The NF4 code book equals the 16 values printed in QLoRA Appendix E to 10<sup>−6</sup>, with an exact zero
  ([arXiv:2305.14314](https://arxiv.org/abs/2305.14314)).
- On normally distributed weights NF4 has lower error than Int4; on uniform weights it has higher error.
- Block constants cost 0.5 bits per parameter, 0.127 with double quantization, which changes the
  reconstruction error by under 1%.

## Reference results

`python lora.py` on our machine (NumPy 2.5):

```text
target: 64 x 48 update of rank 4; full fine-tuning trains 3072 numbers
r =  1:   112 trainable, loss 13.76089, best possible 13.75502
r =  2:   224 trainable, loss 6.59782, best possible 6.58861
r =  4:   448 trainable, loss 0.00000, best possible 0.00000
r =  8:   896 trainable, loss 0.00006, best possible 0.00000
```

## Tasks

1. The LoRA paper initializes B to zero and A at random. What goes wrong if both start at zero? If both
   start random?
2. Change `alpha` so that α/r stays fixed while r grows, then so that α stays fixed. Which keeps the best
   learning rate the same across ranks? (Biderman et al. recommend α = 2r; deck 17.)
3. Make the target update full rank (`--true-rank 48`). Plot the trained loss against r and compare with
   the singular values of the target. This is what "LoRA learns less" means (deck 17).
4. Quantize a random W0 to NF4, then fit a LoRA update that should recover W0 from the quantized weight.
   How much of the quantization error can rank 8 absorb, and why so little?
5. Count the memory to fine-tune a 65B model with QLoRA: 4-bit weights, double-quantized constants, LoRA
   on every linear layer at r = 64 with Adam state in 32 bits. Compare with 16 bytes per parameter for full
   fine-tuning (deck 09).
