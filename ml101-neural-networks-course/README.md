# ML 101: Neural Network Foundations

Learn the machine learning that comes before the Transformer, from linear regression to backpropagation, CNNs, RNNs and attention, with every idea built from scratch in a CPU lab.

**Status, 2026-11-06: complete.** All 21 decks (the course map and Parts I to VI: learning from data, neural networks, optimization and regularization, convolutional networks, sequence models, and practice with the bridge to the Transformer), labs L0 to L6, the syllabus and the bibliography are published.

[Research audit, 2026-11-07](references/research_audit_2026_11.md): checked readings, compatibility findings, prioritized updates and potential projects. Proposed experiments are distinguished from executed results.

## Start here

- [Course map deck](slides/00_course_map.html): the whole course on 9 slides.
- [Syllabus](SYLLABUS.md): parts, weekly plan, labs and assessment.
- [Bibliography](references/bibliography.md): every source, grouped by the deck that reads it.
- [Labs](labs/): CPU-only exercises with their own checks (`python -m unittest` in each lab folder).

## What the course covers

The course follows one training loop (data, model, loss, gradient, update, evaluation) and makes each part of it more capable in turn.

- **Learning from data.** Linear and logistic regression, softmax and cross-entropy, gradient descent and SGD, and the difference between fitting the training set and generalizing.
- **Neural networks.** The perceptron and its limits, multilayer perceptrons, backpropagation as reverse-mode differentiation, and initialization that keeps deep networks trainable.
- **Optimization and regularization.** Momentum, RMSProp, Adam and AdamW, learning-rate schedules and warm-up, weight decay, dropout, early stopping, data augmentation and batch normalization.
- **Convolutional networks.** Convolution and pooling, LeNet, AlexNet, VGG and ResNet.
- **Sequence models.** Recurrent networks and backpropagation through time, LSTM and GRU, sequence-to-sequence models and the attention mechanism of Bahdanau and Luong.
- **Practice and the bridge.** A recipe for debugging training runs, and a map from every idea in the course to the parts of a Transformer block.

Every deck ends with where its idea shows up in today's LLMs, and with check-yourself questions and its lab.

## Audience and prerequisites

Engineers and students who want the foundations before reading LLM papers. You need Python and NumPy, linear algebra (matrix products, transposes, eigenvalues), calculus up to the chain rule, and basic probability. All labs run on a laptop CPU in seconds to minutes; L0 to L4 use only NumPy.

The Transformer and its variants are taught in [LLM Transformer Architecture](../llm-transformer-architecture-course/), which this course leads into. LLM pre-training at scale and post-training (instruction tuning, preference optimization, RLHF) are taught in [LLM Training and Post-Training](../llm-training-course/).

## Slide decks

| Deck | Teaching use |
|---|---|
| [Course map](slides/00_course_map.html) (9 slides) | Week 1: the training loop, the six parts, how each deck is built |
| **Part I. Learning from data** | |
| [01 Linear regression and least squares](slides/01_linear_regression.html) (10 slides) | Week 1: the squared loss and its likelihood, normal equations, gradient descent, ridge; lab L0 |
| [02 Logistic regression and softmax](slides/02_logistic_softmax.html) (10 slides) | Week 1: sigmoid and log-odds, cross-entropy from the likelihood, softmax, stable computation; lab L0 |
| [03 Gradient descent and SGD](slides/03_gradient_descent.html) (10 slides) | Week 2: step size and curvature, mini-batch gradients, the training loop, gradient checks; lab L0 |
| [04 Generalization: overfitting, bias and variance](slides/04_generalization.html) (10 slides) | Week 2: validation sets, the degree sweep, bias-variance, double descent, learning curves; lab L0 |
| **Part II. Neural networks** | |
| [05 From the perceptron to the MLP](slides/05_perceptron_mlp.html) (10 slides) | Week 3: XOR, hidden layers, activations, universal approximation; lab L1 |
| [06 Backpropagation](slides/06_backpropagation.html) (10 slides) | Week 3: computational graphs, reverse mode, layer gradients, an autograd engine; lab L1 |
| [07 Initialization and vanishing gradients](slides/07_initialization.html) (10 slides) | Week 4: variance through a layer, Xavier and He initialization, signals measured by depth; lab L1 |
| **Part III. Optimization and regularization** | |
| [08 Momentum, RMSProp and Adam](slides/08_optimizers.html) (10 slides) | Week 4: momentum and Nesterov, per-parameter steps, Adam's bias correction, AdamW; lab L2 |
| [09 Learning-rate schedules and batch size](slides/09_schedules_batch.html) (10 slides) | Week 5: the noise floor, step and cosine decay, warm-up, linear scaling, gradient clipping; lab L2 |
| [10 Regularization](slides/10_regularization.html) (10 slides) | Week 5: weight decay, early stopping, dropout, data augmentation, label smoothing; lab L3 |
| [11 Batch normalization](slides/11_batchnorm.html) (10 slides) | Week 6: the transform, training against inference, why it works, layer norm; lab L3 |
| **Part IV. Convolutional networks** | |
| [12 Convolutions and LeNet](slides/12_convolutions_lenet.html) (10 slides) | Week 6: local and shared weights, receptive fields, pooling, LeNet-5; lab L4 |
| [13 AlexNet and VGG](slides/13_alexnet_vgg.html) (10 slides) | Week 7: ImageNet 2012, ReLU, dropout and augmentation at scale, stacks of 3 × 3 filters; lab L4 |
| [14 ResNet and residual connections](slides/14_resnet.html) (10 slides) | Week 7: the degradation problem, identity shortcuts, bottlenecks, the residual stream; lab L4 |
| **Part V. Sequence models** | |
| [15 Recurrent networks and backpropagation through time](slides/15_rnn_bptt.html) (10 slides) | Week 8: the recurrence, BPTT, vanishing and exploding gradients, clipping, character language models; lab L5 |
| [16 LSTM and GRU](slides/16_lstm_gru.html) (10 slides) | Week 8: the cell state and gates, the forget gate and its bias, the GRU, which unit to use; lab L5 |
| [17 Sequence to sequence](slides/17_seq2seq.html) (10 slides) | Week 9: the encoder–decoder, teacher forcing, beam search, reversing the source, the bottleneck; lab L6 |
| [18 Attention in sequence to sequence](slides/18_attention_seq2seq.html) (10 slides) | Week 9: Bahdanau and Luong attention, alignments, what the lab's attention learned; lab L6 |
| **Part VI. Practice and the bridge** | |
| [19 Training practice: a debugging recipe](slides/19_debugging_recipe.html) (10 slides) | Week 10: baselines, the loss at initialization, overfitting on purpose, the learning rate, reading curves, fair tuning |
| [20 Bridge to the Transformer](slides/20_bridge_transformer.html) (10 slides) | Week 10: why self-attention replaced recurrence, each part of a Transformer block mapped to its deck, recurrence returning |

## Repository layout

```text
ml101-neural-networks-course/
├── labs/
│   ├── L0_linear_logistic/
│   │   ├── README.md
│   │   ├── linear_models.py         # Least squares, GD and SGD, logistic and softmax regression, degree sweep
│   │   └── test_linear_models.py
│   ├── L1_autograd_mlp/
│   │   ├── README.md
│   │   ├── autograd.py              # Reverse-mode autodiff on NumPy arrays
│   │   ├── gradient_diagnostics.py # Step-size, kink and near-zero checks
│   │   ├── test_gradient_diagnostics.py
│   │   ├── figures-report.json
│   │   ├── figures/
│   │   │   ├── smooth_step.svg
│   │   │   └── near_zero.svg
│   │   ├── mlp.py                   # MLPs, initialization schemes, XOR and spirals, signal scale by depth
│   │   └── test_mlp.py
│   ├── L2_optimizers/
│   │   ├── README.md
│   │   ├── optimizers.py            # SGD, momentum, AdaGrad, RMSProp, Adam, AdamW; schedules; test problems
│   │   └── test_optimizers.py
│   ├── L3_regularization_bn/
│   │   ├── README.md
│   │   ├── regularization.py        # Hand-written MLP with dropout and batch norm; weight decay, early stopping
│   │   └── test_regularization.py
│   ├── L4_cnn_resnet/
│   │   ├── README.md
│   │   ├── cnn.py                   # im2col convolution, pooling, a small CNN, plain against residual depth
│   │   └── test_cnn.py
│   ├── L5_rnn_lstm_gru/
│   │   ├── README.md
│   │   ├── rnn.py                   # RNN, LSTM and GRU cells with BPTT; adding problem; character language model
│   │   └── test_rnn.py
│   ├── L6_seq2seq_attention/
│   │   ├── README.md
│   │   ├── seq2seq.py               # GRU encoder-decoder with no, dot or additive attention; digit reversal
│   │   └── test_seq2seq.py
│   └── README.md                    # Lab index
├── references/
│   ├── research_audit_2026_11.md
│   └── bibliography.md              # Every source, grouped by deck; checked 2026-11-06
├── slides/
│   ├── 00_course_map.html           # Course map deck
│   ├── 01_linear_regression.html
│   ├── 02_logistic_softmax.html
│   ├── 03_gradient_descent.html
│   ├── 04_generalization.html
│   ├── 05_perceptron_mlp.html
│   ├── 06_backpropagation.html
│   ├── 07_initialization.html
│   ├── 08_optimizers.html
│   ├── 09_schedules_batch.html
│   ├── 10_regularization.html
│   ├── 11_batchnorm.html
│   ├── 12_convolutions_lenet.html
│   ├── 13_alexnet_vgg.html
│   ├── 14_resnet.html
│   ├── 15_rnn_bptt.html
│   ├── 16_lstm_gru.html
│   ├── 17_seq2seq.html
│   ├── 18_attention_seq2seq.html
│   ├── 19_debugging_recipe.html
│   └── 20_bridge_transformer.html
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
