# Lab L1: one block, three families

Used with decks 01 to 04. CPU only, NumPy only.

`transformer.py` builds one pre-LN Transformer block and runs it three ways:

| Mode | Mask | Extra sublayer | Family | Deck |
|---|---|---|---|---|
| `encode` | none | none | encoder-only (BERT) | 03 |
| `decode` | causal | none | decoder-only (GPT) | 02 |
| `seq2seq` | causal on the target | cross-attention to the encoder output | encoder-decoder (2017 Transformer, T5) | 01, 04 |

The weights are random and tiny (d = 32, 4 heads, 2 layers): the lab is about the data flow and the masks.

## Run

```bash
python -m unittest test_transformer.py
```

## Tasks

1. Read `attention()` and `naive_attention()` in the test. Explain why the matrix version is the same
   computation, and which line implements the √d<sub>k</sub> scaling from deck 01.
2. `test_decoder_never_sees_the_future` changes the last three tokens. Remove the mask in `decode` and
   watch the test fail. Which deck's model could never be trained without that mask?
3. `test_attention_is_order_blind_without_positions` shows why positions are needed. Remove
   `sinusoidal_positions` from `_inputs` and show that `encode` then returns permuted outputs for permuted inputs.
4. Change `Block` to post-LN, `LayerNorm(x + Sublayer(x))` as in the 2017 paper. Do all tests still pass? Why?
5. Count the parameters of one decoder block and one encoder-decoder block. How much does cross-attention add?
   Relate this to T5's comparison of a 2P-parameter encoder-decoder with a P-parameter decoder (deck 04).

## Checks

Attention matches a one-query-at-a-time loop; the decoder's output at position i does not change when later
tokens change; the encoder's and the encoder-decoder's outputs do; unmasked attention without positions is
permutation-equivariant; and the sinusoidal encoding of position p + k is a rotation of that of p.

## Shared prerequisite bridge

The [foundations worksheet](../../../ml101-neural-networks-course/references/decoder_bridge_worksheet_2026_11.md) links MLP derivative/shape checks to this block and training L1. Its runner reuses all six tests here and temporarily supplies an all-allowed decoder mask: the existing attention-definition test still passes while the exact existing future-invariance test fails. It then restores the mask and verifies the causal check. Original source and tests are unchanged; numerical derivatives and causal data dependence need separate oracles.
