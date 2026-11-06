# ML 101: Neural Network Foundations

Learn the machine learning that comes before the Transformer, from linear regression to backpropagation, CNNs, RNNs and attention, with every idea built from scratch in a CPU lab.

**Status, 2026-11-06: Parts I and II published.** The course map, the four Part I decks on learning from data (linear regression, logistic regression and softmax, gradient descent and SGD, generalization) and the three Part II decks on neural networks (the perceptron and the MLP, backpropagation, initialization), labs L0 and L1, the syllabus and the bibliography are in place. Parts III to VI and labs L2 to L6 follow in later pull requests.

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

Engineers and students who want the foundations before reading LLM papers. You need Python and NumPy, linear algebra (matrix products, transposes, eigenvalues), calculus up to the chain rule, and basic probability. All labs run on a laptop CPU in seconds to minutes; L0 to L5 use only NumPy.

The Transformer and its variants are taught in [LLM Transformer Architecture](../llm-transformer-architecture-course/), which this course leads into. LLM pre-training at scale and post-training (instruction tuning, preference optimization, RLHF) are left to a separate course.

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

Parts III to VI (decks 08 to 20) are listed in the [syllabus](SYLLABUS.md) and added with their labs.

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
│   │   ├── mlp.py                   # MLPs, initialization schemes, XOR and spirals, signal scale by depth
│   │   └── test_mlp.py
│   └── README.md                    # Lab index
├── references/
│   └── bibliography.md              # Every source, grouped by deck; checked 2026-11-06
├── slides/
│   ├── 00_course_map.html           # Course map deck
│   ├── 01_linear_regression.html
│   ├── 02_logistic_softmax.html
│   ├── 03_gradient_descent.html
│   ├── 04_generalization.html
│   ├── 05_perceptron_mlp.html
│   ├── 06_backpropagation.html
│   └── 07_initialization.html
├── LICENSE                          # Apache-2.0
├── README.md                        # This file
└── SYLLABUS.md                      # Parts, weekly plan, labs and assessment
```

## License

Apache-2.0, see [LICENSE](LICENSE).
