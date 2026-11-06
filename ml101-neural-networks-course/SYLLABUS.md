# Syllabus: ML 101, neural network foundations

Ten weeks, two to three decks a week, with a lab every week. The course assumes Python, NumPy, linear algebra,
the chain rule and basic probability, and nothing about neural networks. It ends where the
[LLM Transformer Architecture](../llm-transformer-architecture-course/) course begins.

Each deck follows six steps: the **idea** (the problem it solves), the **mechanics** (the equations), a **worked
example** with numbers measured in the course's labs, the **traps** (what goes wrong and how to see it),
**where it shows up in LLMs**, and **check yourself** questions with the matching lab. Sources are listed in the
[bibliography](references/bibliography.md).

**Status, 2026-11-06:** all 21 decks (00 to 20) and labs L0 to L6 are published.

## Parts and decks

| Part | Deck | Topic | Main sources |
|---|---|---|---|
| 0 Map | 00 | Course map | |
| I Learning from data | 01 | Linear regression and least squares | Hastie et al., ch. 3 |
| | 02 | Logistic regression and softmax | Hastie et al., ch. 4; Goodfellow et al., §6.2 |
| | 03 | Gradient descent and SGD | Robbins and Monro 1951; Bottou et al. 2016 |
| | 04 | Generalization: overfitting, bias and variance | Hastie et al., ch. 7; Zhang et al. 2016; Belkin et al. 2018; Nakkiran et al. 2019 |
| II Neural networks | 05 | From the perceptron to the MLP | Rosenblatt 1958; Cybenko 1989; Glorot et al. 2011 |
| | 06 | Backpropagation | Rumelhart et al. 1986; Baydin et al. 2015 |
| | 07 | Initialization and vanishing gradients | Glorot and Bengio 2010; He et al. 2015 |
| III Optimization and regularization | 08 | Momentum, RMSProp and Adam (with AdamW) | Sutskever et al. 2013; Duchi et al. 2011; Kingma and Ba 2014; Loshchilov and Hutter 2017 |
| | 09 | Learning-rate schedules and batch size | Loshchilov and Hutter 2016; Goyal et al. 2017; McCandlish et al. 2018 |
| | 10 | Regularization: weight decay, dropout, early stopping, augmentation | Srivastava et al. 2014; Krizhevsky et al. 2012 |
| | 11 | Batch normalization | Ioffe and Szegedy 2015; Santurkar et al. 2018; Ba et al. 2016 |
| IV Convolutional networks | 12 | Convolutions and LeNet | LeCun et al. 1989 and 1998 |
| | 13 | AlexNet and VGG | Krizhevsky et al. 2012; Simonyan and Zisserman 2014 |
| | 14 | ResNet and residual connections | He et al. 2015 and 2016; Srivastava et al. 2015 |
| V Sequence models | 15 | Recurrent networks and backpropagation through time | Elman 1990; Werbos 1990; Bengio et al. 1994; Pascanu et al. 2012 |
| | 16 | LSTM and GRU | Hochreiter and Schmidhuber 1997; Gers et al. 2000; Cho et al. 2014; Greff et al. 2015 |
| | 17 | Sequence to sequence | Sutskever et al. 2014; Cho et al. 2014 |
| | 18 | Attention in sequence to sequence (Bahdanau, Luong) | Bahdanau et al. 2014; Luong et al. 2015 |
| VI Practice and the bridge | 19 | Training practice: a debugging recipe | Karpathy 2019; Godbole et al. 2023; Bergstra and Bengio 2012 |
| | 20 | Bridge to the Transformer | Vaswani et al. 2017; Katharopoulos et al. 2020 |

## Weekly plan

| Week | Decks | Lab |
|---|---|---|
| 1 | 00, 01, 02 | L0: least squares, gradient descent, logistic and softmax regression |
| 2 | 03, 04 | L0: SGD, gradient checks, the degree sweep and ridge |
| 3 | 05, 06 | L1: the autograd engine; MLPs on XOR and spirals |
| 4 | 07, 08 | L1: initialization; L2: optimizers |
| 5 | 09, 10 | L2: schedules; L3: regularization |
| 6 | 11, 12 | L3: batch normalization; L4: convolution by hand |
| 7 | 13, 14 | L4: a small CNN and residual blocks |
| 8 | 15, 16 | L5: RNN, LSTM and GRU |
| 9 | 17, 18 | L6: seq2seq with and without attention |
| 10 | 19, 20 | Final project |

## Labs

All labs run on a laptop CPU in seconds to minutes and check themselves with `python -m unittest`.

| Lab | Task | Check |
|---|---|---|
| L0 | Linear, logistic and softmax regression from scratch | Closed form against gradient descent; every gradient against finite differences; U-shaped validation error |
| L1 | A reverse-mode autograd engine and MLPs | Finite differences for every parameter; XOR and spirals solved; activation scale per layer for each initialization |
| L2 | SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW and schedules | Bias correction and decoupled decay by construction; momentum beats GD on a narrow quadratic; decaying schedules beat a constant rate |
| L3 | Weight decay, dropout, early stopping and batch normalization | Batch norm gradients by finite differences; a memorizing network made to generalize; batch norm across a 30-fold learning-rate range |
| L4 | Convolution and pooling by hand, a small CNN, residual blocks | im2col against a direct loop; CNN against MLP on moved shapes; a 64-layer residual network trains where a plain one cannot |
| L5 | RNN, LSTM and GRU with backpropagation through time | Every gradient by finite differences; gradient reaching early steps against the recurrent scale; LSTM and GRU solve the adding problem at length 100 where an RNN is at chance |
| L6 | Seq2seq with and without attention | Gradients of all three models by finite differences; attention keeps accuracy as strings get longer; attention weights follow the anti-diagonal |

## Assessment

- **Labs (50%)**: each lab's tasks, with the numbers you measured and a sentence on what they show.
- **Check-yourself questions (20%)**: written answers for one deck per week.
- **Final project (30%)**: take one dataset (a small image, text or tabular set that fits on a CPU), train a
  linear baseline, an MLP and a CNN or RNN, and report a learning-rate sweep, a regularization comparison and
  validation curves. Grading is on whether the comparisons are fair (same data, budget and seeds) and whether
  the conclusions follow from the curves.
