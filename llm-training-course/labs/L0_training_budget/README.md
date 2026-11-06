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
  - `pipeline_bubble` and `bubble_over_ideal`: the idle share of a pipeline step, plain and interleaved (deck 09).
  - `recompute_days`: Megatron's 8TP / (nX) estimate of training time with activation recomputation (deck 09).
  - `activation_bytes`: activation memory per layer with tensor and sequence parallelism and selective
    recomputation (deck 09).
- `test_budget.py`: the checks.

## Run

```bash
python budget.py --params 405e9 --tokens 15.6e12 --gpus 16384 --peak 989e12 --mfu 0.40
python budget.py --compute 5.88e23
python budget.py --params 7.5e9 --dp 64
python budget.py --seq 2048 --hidden 12288 --heads 96 --tp 8
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

- GPipe's bubble is under 20% of a step at M = 4K micro-batches, and the interleaved schedule divides
  Megatron's bubble by v ([arXiv:1811.06965](https://arxiv.org/abs/1811.06965) §2.3,
  [arXiv:2104.04473](https://arxiv.org/abs/2104.04473) §2.2).
- 8TP / (nX) gives the 34 days for GPT-3 175B and 84 days for the 1T model in
  [arXiv:2104.04473](https://arxiv.org/abs/2104.04473) §5.1.
- Activation memory follows equations 1, 2 and 4 and Table 2 of
  [arXiv:2205.05198](https://arxiv.org/abs/2205.05198).

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
6. For a GPT-3-sized layer (s = 2048, h = 12288, a = 96, micro-batch 1, t = 8), how much of the activation
   memory is the 5as/h attention term? What does selective recomputation give up to remove it?
7. A 16-stage pipeline needs its bubble under 10% of the step. How many micro-batches does GPipe need,
   and how many does an interleaved schedule with v = 4 need? What does a larger v cost in communication?
