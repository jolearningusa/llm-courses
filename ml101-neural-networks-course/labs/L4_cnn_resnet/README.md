# Lab L4: convolution, pooling and residual connections

Used with decks 12 to 14. CPU only, NumPy only.

`cnn.py` contains:

| Part | Names | Deck |
|---|---|---|
| Shapes and counts | `conv_out_size`, `conv_params`, `receptive_field` | 12, 13 |
| Layers (forward and backward) | `Conv2d` (im2col), `MaxPool2d`, `GlobalAvgPool`, `ReLU`, `Flatten`, `Linear`, `Sequential`, `Residual` | 12, 14 |
| Data | `shapes` (four 5 × 5 shapes at random positions in noisy 12 × 12 images), `spirals` | 12, 14 |
| Experiments | `small_cnn` against `small_mlp` on moved shapes; `depth_experiment` (plain against residual MLPs of depth 8, 32 and 64) | 12, 14 |

## Run

```bash
python cnn.py                # the numbers quoted in decks 12 to 14 (about 30 s)
python -m unittest           # 10 checks, about 20 s
```

## Tasks

1. Work out by hand the shapes in `im2col` for a batch of 2 RGB 7 × 7 images, k = 3, stride 2, padding 1.
2. Replace max pooling with a stride-2 convolution in `small_cnn`. Compare accuracy and parameters.
3. Remove `GlobalAvgPool` and use `Flatten` plus `Linear` instead. What happens to accuracy on moved shapes, and why?
4. Give the MLP 10 times more hidden units. Does it ever generalize to moved shapes?
5. In `depth_experiment`, remove the 1/√depth scaling of the residual branch. Then try batch norm in the plain
   network (you can import `BatchNorm` from lab L3). How deep can each go?
6. Visualize the 8 filters of the CNN's first layer after training. What do they detect?

## Checks

Output sizes match AlexNet, LeNet-5 and "same" padding; receptive fields of stacked 3 × 3 layers are 5 and 7;
three 3 × 3 layers have fewer weights than one 7 × 7; the im2col convolution equals a direct loop for several
strides and paddings; convolution gradients match finite differences; max pooling routes the gradient to the
maximum; convolution is translation equivariant; the whole CNN's first-layer gradient matches finite differences;
a CNN generalizes to moved shapes while an MLP of the same size does not; and at 64 layers a residual network
trains while a plain one does not.
