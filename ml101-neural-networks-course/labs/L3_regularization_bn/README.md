# Lab L3: regularization and batch normalization

Used with decks 10 and 11. CPU only, NumPy only.

`regularization.py` builds an MLP from layers with hand-written `forward` and `backward`: `Linear`, `ReLU`,
`Dropout` (inverted) and `BatchNorm` (training and evaluation modes, running statistics).

| Experiment | What it shows | Deck |
|---|---|---|
| `regularization_table` | A 67,000-parameter MLP on 40 noisy points: no regularization, weight decay 10<sup>−2</sup>, dropout 0.5; final and best-checkpoint (early stopping) validation loss and accuracy | 10 |
| `batchnorm_lr_sweep` | An 8-hidden-layer MLP trained at learning rates 0.03, 0.3 and 1.0, with and without batch norm | 11 |

## Run

```bash
python regularization.py     # the numbers quoted in decks 10 and 11 (about 20 s)
python -m unittest           # 11 checks, about 15 s
```

## Tasks

1. Derive the backward pass of `BatchNorm` (the two correction terms) on paper and compare it with the code.
2. Sweep the weight decay from 10<sup>−4</sup> to 10<sup>−1</sup> and plot final validation loss against it.
   Where does it start to underfit?
3. Combine dropout 0.5 with early stopping and with weight decay. Which combination has the best
   validation loss?
4. Grow the training set from 40 to 400 points and rerun `regularization_table`. How much does each
   regularizer still help?
5. Train with batch norm and evaluate in training mode by mistake (`train=True`). How do the predictions
   change with the batch size used for evaluation?
6. Replace `BatchNorm` by a layer norm (statistics over features for each example) and rerun the learning-rate
   sweep.

## Checks

Batch norm standardizes in training, matches finite differences, and in evaluation mode gives an example the
same output alone or in a batch; inverted dropout keeps the expected activation and is the identity at
evaluation; every gradient of a network with batch norm matches finite differences; weight decay with SGD is an
L2 penalty; the unregularized network memorizes the flipped labels and its validation loss doubles after its best
point; weight decay halves the final validation loss and stops it fitting every flipped label; dropout's best
checkpoint beats the unregularized one; and batch norm trains at every learning rate in the sweep while the plain
network diverges at the largest.
