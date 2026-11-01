# Lab L0: what a config.json implies

Used with decks 01, 05 and 06. CPU only, standard library only.

A model's `config.json` fixes its parameter count, its training cost per token and the size of its KV cache.
This lab computes all three from the config alone and checks the parameter count against the released
checkpoints.

## Files

- `config_math.py`: loads a config (GPT-2 or the Llama family: Llama, Mistral, Qwen2) and computes parameters
  by component, KV-cache bytes per token and the 6ND training-compute estimate.
- `configs/`: five released configs, copied from the model repositories on Hugging Face on 2026-11-01
  (`openai-community/gpt2`, `NousResearch/Llama-2-7b-hf` and `NousResearch/Meta-Llama-3-8B` as ungated
  copies of Meta's files, `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-7B`). Fields the lab does not use,
  such as `transformers_version`, were dropped.
- `test_config_math.py`: the checks.

## Run

```bash
python config_math.py configs/*.json
python config_math.py configs/llama-3-8b.json --tokens 15e12   # adds 6ND and tokens per parameter
python -m unittest test_config_math.py
```

## Tasks

1. Read `params()`. For each line, name the weight matrix it counts and the deck that introduced it
   (attention projections in deck 01, tied embeddings in deck 02, the SwiGLU gate in deck 18).
2. Explain why GPT-2 has a `positions` term and the Llama family does not.
3. Llama 2 7B and Llama 3 8B have the same width and depth. Account for the 1.29B difference in total
   parameters: how much comes from the vocabulary, and how much from the FFN width (11,008 against 14,336)?
4. The KV cache per token is 2 × layers × KV heads × head dimension × bytes. Compute it for each config at
   its maximum context. Why is Llama 3 8B's cache a quarter of Llama 2 7B's per token?
5. Add a config of your choice from a model card (or write one for GPT-3 175B from deck 05) and check your
   total against the published parameter count.

## Checks

The computed totals must equal the parameter counts of the released weights. Qwen2.5-7B's model card gives
7.61B total and 6.53B non-embedding parameters, which the test also checks.
