# Lab L4: chat templates and packing masks

Used with deck 11. CPU only, NumPy only.

`chat.py` renders a conversation into the single text stream a decoder reads, using two real formats
written out by hand: Llama 3 Instruct (header tokens and `<|eot_id|>`) and Qwen2.5 Instruct (ChatML-style
`<|im_start|>` / `<|im_end|>`, with a default system message). It then builds the training-side masks:
document ids for packed sequences, the document mask Llama 3 uses to stop attention across packed
documents, position ids that restart per document, and a loss mask over assistant replies only.

## Run

```bash
python -m unittest test_chat.py
```

## Tasks

1. Render the same three-turn chat with both templates. Count the special tokens each one adds per turn.
2. Remove the system message. What does each template do? Why might a default system message change a
   model's behavior?
3. `segments()` marks which text the loss is computed on. Why is the end-of-turn marker trained, but the
   assistant header is not?
4. Pack documents of lengths 5, 3 and 8. Compare the causal mask and the document mask: how many query-key
   pairs does the document mask remove? What would a model learn if they were left in?
5. With `positions_restart()`, each packed document starts at position 0. What would RoPE (deck 08) see if
   positions did not restart?

## Checks

Both renderings equal strings written from the published formats (Meta's Llama 3 prompt-format page and the
`chat_template` in Qwen2.5-7B-Instruct's `tokenizer_config.json`); Qwen inserts its default system message;
the loss mask covers exactly the assistant replies and their end-of-turn markers; the document mask removes
all 21 cross-document pairs for documents of 3, 2 and 3 tokens; and positions restart per document.

The lab does not load the tokenizers, so it checks the text format rather than token IDs. With the
`transformers` package and network access, `tokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True)` gives the reference string to compare against.
