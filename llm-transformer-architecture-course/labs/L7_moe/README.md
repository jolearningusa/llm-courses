# Lab L7: a top-k MoE layer and load balancing

Used with decks 19 to 22. CPU only, NumPy only.

`moe.py` builds one mixture-of-experts layer and the tools to keep it balanced:

| Code | Idea | Deck |
|---|---|---|
| `MoE(n_experts, k)` | Router, softmax over the top-k logits, SwiGLU experts | 19, 21 |
| `MoE(capacity_factor=...)` | Expert capacity; tokens over capacity are dropped (they skip the layer) | 19, 20 |
| `MoE(n_shared=...)` | Shared experts that process every token | 22 |
| `switch_aux_loss()` | α · N · Σ f<sub>i</sub> P<sub>i</sub> | 20 |
| `aux_loss_grad_step()` | One gradient step on the router for that loss | 20 |
| `LossFreeBalancer` | Per-expert bias for choosing experts, raised for underloaded and lowered for overloaded experts | 22 |
| `max_violation()` | MaxVio = (max load − mean load) / mean load | 22 |
| `moe_params()` | Total and active parameters of a Mixtral-style model | 21 |

## Run

```bash
python -m unittest test_moe.py
python moe.py
```

`python moe.py` routes a stream of 300 skewed batches (256 tokens, 8 experts, top-2) three ways. Our run:

| Routing | MaxVio, last 20 batches |
|---|---|
| Untrained router | 1.275 |
| Router trained on the Switch loss only (loss 1.318 → 1.004) | 0.181 |
| Loss-free bias, u = 0.01 | 0.201 |

It also prints Mixtral 8x7B's size from its Table 1 shape: 46.70B total and 12.88B active parameters, the
paper's "47B" and "13B".

The tests check that top-k with k = N equals a dense mixture; that gates sum to one; that capacity is
respected and dropped tokens are counted; that the Switch loss is 1 under uniform routing and N when
collapsed; that the loss-free bias changes which experts are chosen but not the gate values; that both
balancing methods cut MaxVio by more than 3×; and the Mixtral and C(64, 8) numbers on the decks.

## Tasks

1. Plot the expert loads per batch (a heatmap of `info["counts"]`) for the three routings in `main()`.
2. Set `capacity_factor` to 1.0, 1.25 and 2.0 for the untrained router and count dropped tokens. Which
   tokens are dropped first in `forward`, and why is that a poor rule for a real model?
3. The demo uses u = 0.01, ten times the paper's 0.001, because its router logits are on a different
   scale and the run is short. Try u = 0.001 and u = 0.1. Which converges, which oscillates (deck 22)?
4. `aux_loss_grad_step` trains the router on the balance loss alone, so balance is free. In a real model
   the router also gets language-modeling gradients. Explain in two sentences why that is where the
   auxiliary loss starts to hurt quality, and why the bias method avoids it.
5. Split each of 8 experts into 4 (32 experts, `d_ff` ÷ 4, k = 8) and add 1 shared expert. Check the
   active expert parameters equal the 8-expert top-2 layer's (DeepSeekMoE, deck 22).
6. Use `moe_params` on a DeepSeek-V3 or Qwen3 MoE config. What does it miss (MLA, shared experts,
   dense first layers), and how would you extend it?
