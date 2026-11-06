# Labs

CPU-only labs, each checked against a closed form, finite differences or a measured property. The
[syllabus](../SYLLABUS.md) lists all seven.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_linear_logistic/) | Linear, logistic and softmax regression from scratch; gradient checks; overfitting and ridge | 01–04 | Python 3.10+, NumPy |
| [L1](L1_autograd_mlp/) | A reverse-mode autograd engine; MLPs on XOR and spirals; initialization and signal scale | 05–07 | Python 3.10+, NumPy |
| [L2](L2_optimizers/) | SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW; learning-rate schedules | 08–09 | Python 3.10+, NumPy |
| [L3](L3_regularization_bn/) | Weight decay, dropout, early stopping and batch normalization in a hand-written MLP | 10–11 | Python 3.10+, NumPy |
| [L4](L4_cnn_resnet/) | Convolution and pooling with forward and backward passes; a small CNN; plain against residual depth | 12–14 | Python 3.10+, NumPy |
| [L5](L5_rnn_lstm_gru/) | RNN, LSTM and GRU with backpropagation through time; the adding problem; a character language model | 15–16 | Python 3.10+, NumPy |
| [L6](L6_seq2seq_attention/) | A GRU encoder–decoder with and without attention, on reversing digit strings | 17–18, 20 | Python 3.10+, NumPy; imports the cells from L5 |

Run every lab's checks from the lab directory with `python -m unittest`.
