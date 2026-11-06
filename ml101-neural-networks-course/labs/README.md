# Labs

CPU-only labs, each checked against a closed form, finite differences or a measured property. The
[syllabus](../SYLLABUS.md) lists all seven; L0 and L1 are published, L2 to L6 follow with their decks.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_linear_logistic/) | Linear, logistic and softmax regression from scratch; gradient checks; overfitting and ridge | 01–04 | Python 3.10+, NumPy |
| [L1](L1_autograd_mlp/) | A reverse-mode autograd engine; MLPs on XOR and spirals; initialization and signal scale | 05–07 | Python 3.10+, NumPy |

Run every lab's checks from the lab directory with `python -m unittest`.
