# Lab L6: sequence to sequence with and without attention

Used with decks 17, 18 and 20. CPU only, NumPy only. It imports the GRU cell, Adam and the loss from
[lab L5](../L5_rnn_lstm_gru/), so keep the two folders side by side.

`seq2seq.py` contains:

| Part | Names | Deck |
|---|---|---|
| Model | `Seq2Seq(attention)` with attention `None`, `"dot"` or `"additive"`: embeddings, a GRU encoder and decoder, attention, a linear output layer; forward with teacher forcing, backward by hand | 17, 18 |
| Decoding | `greedy` (optionally returns the attention weights) | 17 |
| Data | `reverse_batch`: random digit strings and their reversals | 17 |
| Experiments | `train`, `accuracy_by_length`, `mirror_band` (where the attention looks), `attention_text` | 17, 18 |

## Run

```bash
python seq2seq.py            # the numbers quoted in decks 17 and 18 (about three minutes)
python -m unittest           # 8 checks, about 20 s
```

## Tasks

1. Write down the shapes of every array in `Seq2Seq.backward` for a batch of B strings of length T.
2. Train without attention on lengths 4 to 12 only. Does it reach 100% at length 12? What is the shortest length
   at which it fails with hidden size 32, 64 and 128?
3. Reverse the source before encoding (Sutskever et al.'s trick) in the model without attention. Since the target is
   the reversed source, what does the task become, and why does it get much easier?
4. Change the task to copying (target = source). Where does the additive attention look now?
5. Add a backward GRU to the encoder and concatenate the two states, as Bahdanau et al. did. Does the attention
   peak move from the digit just written to the digit being written?
6. Train on lengths 4 to 16 and test on 20 and 24. Which model extrapolates to longer strings?

## Checks

Every gradient of the models without attention, with dot attention and with additive attention matches finite
differences; decoder inputs are the target shifted right after a start token; attention weights are a distribution
over input positions; greedy decoding agrees with teacher forcing on its own outputs; after short training,
attention beats a fixed context by more than 20 points of digit accuracy at length 16; and most attention weight
lies within one step of the mirrored input position.
