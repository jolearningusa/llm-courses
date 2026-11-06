# Lab L5: RNN, LSTM and GRU with backpropagation through time

Used with decks 15 and 16. CPU only, NumPy only.

`rnn.py` contains:

| Part | Names | Deck |
|---|---|---|
| Cells (one step forward and backward) | `RNNCell`, `LSTMCell` (forget-gate bias 1), `GRUCell` (as in Cho et al. 2014) | 15, 16 |
| Network | `RecurrentNet`: a cell unrolled over time with a linear read-out on the last step or every step; `backward` is BPTT and records ‖∂L/∂h<sub>t</sub>‖ for each step | 15 |
| Training | `mse`, `softmax_xent`, `clip_by_global_norm`, `Adam` | 15 |
| Experiments | `gradient_decay` (gradient reaching each step at initialization), `gradient_vs_scale` (vanishing and exploding against the recurrent matrix's scale), `adding_problem` and `train_adding`, `char_lm` and `sample` | 15, 16 |

## Run

```bash
python rnn.py                # the numbers quoted in decks 15 and 16 (about two minutes)
python -m unittest           # 11 checks, about 10 s
```

## Tasks

1. Derive `LSTMCell.back` on paper from the forward equations, then compare with the code.
2. Set the LSTM's forget-gate bias to 0 and to 3. How does `gradient_decay` change, and does the adding problem at
   T = 100 still train?
3. Give the GRU an update-gate bias of +2 (so z starts near 0.9). Re-run `gradient_decay` for the GRU.
4. Train the adding problem without gradient clipping (`clip=None`) and plot the gradient norms that `train_adding`
   returns. Then try plain SGD instead of Adam. When does clipping matter?
5. Replace the RNN's orthogonal initialization with a Gaussian of standard deviation 1/√H. What happens to the
   largest eigenvalue of W<sub>h</sub>, and to training at T = 50?
6. Train `char_lm` on a longer text of your own (a few thousand characters). Compare the samples at temperatures
   0.5, 1 and 1.5, and the training loss of the RNN, LSTM and GRU.

## Checks

Every parameter and input gradient of all three cells, for read-out on the last step and on every step, matches
finite differences; the cross-entropy gradient matches too; an open forget gate with a closed input gate keeps
the cell state exactly; global-norm clipping rescales to the threshold; the RNN's gradient vanishes at recurrent
scale 0.5 and explodes at 2; at initialization the LSTM keeps over 100 times more gradient at the first step than
the RNN; and on the adding problem at length 50 an LSTM learns while an RNN stays near chance.
