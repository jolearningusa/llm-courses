# Lab L3: SFT with response-only loss and packing

Used with decks 13 to 15. CPU only, PyTorch 2 (`pip install torch`).

Supervised fine-tuning reuses the pre-training loss, but where that loss is taken and how examples are
batched decide what the model learns and how much compute goes to padding. This lab renders
conversations with a chat template, trains only on the assistant's tokens, packs several conversations
into one row, and checks each step against the unpacked, unmasked computation.

| File | What it holds | Deck |
|---|---|---|
| `chat.py` | A character-level chat template (`<\|user\|>`, `<\|assistant\|>`, `<\|end\|>`) that returns token ids and a loss mask; a synthetic instruction set (reverse, upper, sort, first letter) | 13, 15 |
| `packing.py` | Padded batches and packed batches: document ids, positions that restart per example, and a block-diagonal causal mask | 15 |
| `model.py` | A 2-layer decoder that accepts positions and an attention mask; the masked loss as a sum and a token count | 15 |
| `sft.py` | Trains four variants on the same data, steps and seeds and reports held-out exact match | 13–15 |

The template follows [architecture deck 11](../../../llm-transformer-architecture-course/slides/11_conversation_to_tokens.html):
the loss covers each assistant reply and the `<|end|>` that closes it, so the model learns to stop, but
not the role markers that the template always writes.

## Run

```bash
python -m unittest test_sft.py     # about 5 seconds
python sft.py                      # 4 runs x 3 seeds x 500 steps, about a minute on 4 cores
python sft.py --runs packed --steps 1000
```

## Checks

- The loss mask covers exactly the assistant replies and their closing `<|end|>`.
- Tokens outside the mask get exactly zero gradient.
- A packed batch with a block-diagonal mask and restarted positions gives the same summed loss over the
  same number of tokens as one example per row, with fewer rows; without the block-diagonal mask it does not.
- Packing keeps every token and every trained token.
- Summing the loss over micro-batches and dividing by the total token count equals the full-batch mean;
  averaging each micro-batch's mean does not, the aggregation issue Tülu 3 reports
  ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124) §4.3.2).
- 250 steps of SFT lift held-out exact match from about 0 to above 30%.

## Reference results

Our run on a 4-core CPU with PyTorch 2.15, 500 steps of 32 conversations, AdamW at 3e-3, three seeds.
Your numbers will differ with the PyTorch version and the hardware.

| Run | Exact match | Padding share |
|---|---|---|
| padded (one example per row) | 0.898 ± 0.013 | 39.9% |
| packed (block-diagonal mask) | 0.865 ± 0.025 | 15.3% |
| all-tokens (loss on user tokens too) | 0.292 ± 0.112 | 15.3% |
| packed-naive (plain causal mask, running positions) | 0.170 ± 0.083 | 15.3% |

Padded and packed runs compute the same loss at every step (the checks above). Their trained models still
differ a little, because the two batch layouts round differently and small differences grow over 500 steps.

## Tasks

1. Why does putting the loss on the user's tokens hurt here so much? The user's words are random letters:
   what is the model asked to predict there, and what share of its gradient does that take?
2. The naive packed run trains on rows where an example can attend to the ones before it. At evaluation
   there are none. Describe the mismatch, and say which of the two fixes (the mask or the positions)
   matters more by adding a run with only one of them.
3. Change the micro-batch split in `test_sum_then_divide_matches_full_batch` so the mean of means is
   further from the true mean. What property of the split controls the gap?
4. Make every conversation two turns long and train with the loss on the last reply only. How does
   held-out accuracy on single-turn prompts change?
5. Measure tokens per second for `padded` and `packed` at row lengths of 64, 128 and 256. Where does packing
   stop helping on a CPU, and why would that differ on a GPU with a variable-length attention kernel?
