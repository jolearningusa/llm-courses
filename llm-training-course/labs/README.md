# Labs

CPU-only labs, each checked against a reference value, a closed form or a definition. The
[syllabus](../SYLLABUS.md) lists all ten; labs are added with the part of the course they belong to.

| Lab | Topic | Decks | Needs |
|---|---|---|---|
| [L0](L0_training_budget/) | A training budget: FLOPs, Chinchilla tokens, days at an MFU, ZeRO memory, batch-size trade-off | 01, 02, 04, 09 | Python 3.10+ |
| [L1](L1_pretraining_loop/) | A tiny pre-training loop: hand-written AdamW and Muon, constant, cosine and WSD schedules, QK-norm and z-loss | 01–04 | Python 3.10+, PyTorch 2 (CPU) |

Run every lab's checks from the lab directory with `python -m unittest`.
