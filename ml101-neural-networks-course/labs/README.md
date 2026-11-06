# Labs

CPU-only labs, each checked against a closed form, finite differences or a measured property. The
[syllabus](../SYLLABUS.md) lists all seven; L0 to L4 are published, L5 and L6 follow with their decks.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_linear_logistic/) | Linear, logistic and softmax regression from scratch; gradient checks; overfitting and ridge | 01–04 | Python 3.10+, NumPy |
| [L1](L1_autograd_mlp/) | A reverse-mode autograd engine; MLPs on XOR and spirals; initialization and signal scale | 05–07 | Python 3.10+, NumPy |
| [L2](L2_optimizers/) | SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW; learning-rate schedules | 08–09 | Python 3.10+, NumPy |
| [L3](L3_regularization_bn/) | Weight decay, dropout, early stopping and batch normalization in a hand-written MLP | 10–11 | Python 3.10+, NumPy |
| [L4](L4_cnn_resnet/) | Convolution and pooling with forward and backward passes; a small CNN; plain against residual depth | 12–14 | Python 3.10+, NumPy |

Run every lab's checks from the lab directory with `python -m unittest`.
