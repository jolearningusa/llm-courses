# Lab L0: a training budget

Used with decks 01, 02, 04 and 09. CPU only, standard library only.

Before a run starts, four numbers fix most of its cost: parameters, tokens, GPUs and how fast they
are used. This lab turns them into compute, wall-clock days and memory per GPU, using the formulas the
papers in Part I and Part III use, and checks each against a number a paper published.

## Files

- `budget.py`: the formulas, each with the deck and paper it comes from.
  - `train_flops`: 6ND, about 6 FLOPs per parameter per token for forward and backward
    (architecture course [deck 05](../../../llm-transformer-architecture-course/slides/05_gpt3_scaling_laws.html)).
  - `chinchilla_optimal`: split a FLOP budget at 20 tokens per parameter
    ([deck 06](../../../llm-transformer-architecture-course/slides/06_chinchilla.html) of the architecture course).
  - `train_days` and `mfu`: wall-clock time at a model FLOPs utilization, as defined in PaLM.
  - `model_state_bytes`: weights, gradients and Adam state per GPU at ZeRO stages 0 to 3.
  - `steps_and_examples`: the steps-against-examples trade-off around the critical batch size (deck 02).
  - `muon_overhead`: the extra FLOPs of Muon's Newton-Schulz steps (deck 04).
- `test_budget.py`: the checks.

## Run

```bash
python budget.py --params 405e9 --tokens 15.6e12 --gpus 16384 --peak 989e12 --mfu 0.40
python budget.py --compute 5.88e23
python budget.py --params 7.5e9 --dp 64
python -m unittest test_budget.py
```

`--peak` is the GPU's peak FLOP/s at the training precision, from its datasheet; 989e12 is an example
value for a current data-center GPU in BF16 without sparsity.

## Checks

- 6ND for Llama 3 405B on 15.6T tokens is within 1% of the 3.8 × 10^25 FLOPs the report gives
  ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783) §1).
- Chinchilla's own budget splits back into 70B parameters and 1.4T tokens.
- Model-state memory for 7.5B parameters on 64 data-parallel GPUs reproduces ZeRO's Figure 1: 120, 31.4,
  16.6 and 1.88 GB ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).
- At the critical batch size a run takes twice the minimum steps and twice the minimum examples
  ([arXiv:1812.06162](https://arxiv.org/abs/1812.06162) eq. 2.11).
- Muon's overhead is about 0.7% for the NanoGPT speedrun and 0.5% at Llama 405B's width and batch, as
  in [Jordan (2024)](https://kellerjordan.github.io/posts/muon/).

## Tasks

1. Llama 3 405B trained on 15.6T tokens. How many tokens per parameter is that, and how many times the
   Chinchilla-optimal count? Explain why a model meant to be served many times is trained past the optimum.
2. The Llama 3 report gives 38 to 43% MFU. At 40% on 16,384 GPUs, how many days does 6ND take? Compare
   with the report's 54-day snapshot of 466 interruptions, and say what else costs wall-clock time.
3. For a 70B model on 512 GPUs, find the smallest ZeRO stage at which weights, gradients and Adam state
   fit in 80 GB with 30 GB left for activations.
4. With S_min = 10,000 steps and E_min = 10^10 tokens, what does a batch of 4M tokens cost in steps and
   tokens? Of 16M? Which would you pick with a fixed deadline, and which with a fixed GPU budget?
5. Add a function for the 6ND of a mixture-of-experts model, where N counts only the active parameters.
