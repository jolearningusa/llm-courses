"""Hinton-style distillation with a temperature (deck 29).

The student is trained on the teacher's softened distribution softmax(v / T). The gradient of the
cross-entropy with respect to the student's logits z is (q - p) / T, so its size falls as 1/T^2 once the
probabilities themselves flatten; Hinton et al. multiply the soft loss by T^2 to keep it comparable with the
hard-label loss. At high temperature the gradient approaches (z - v) / (N T^2): matching logits.
"""
import numpy as np


def softmax(x, T=1.0):
    z = x / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def soft_ce(z, v, T):
    """Cross-entropy of student logits z against the teacher's soft targets at temperature T."""
    p = softmax(v, T)
    logq = np.log(softmax(z, T))
    return -(p * logq).sum(-1)


def soft_ce_grad(z, v, T):
    """d soft_ce / dz = (q - p) / T."""
    return (softmax(z, T) - softmax(v, T)) / T


def kd_loss(z, v, label, T=4.0, alpha=0.5):
    """alpha * T^2 * soft cross-entropy + (1 - alpha) * hard cross-entropy at T = 1."""
    hard = -np.log(softmax(z)[label])
    return alpha * T * T * soft_ce(z, v, T) + (1 - alpha) * hard


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    v = rng.normal(0, 3, size=10)
    v -= v.mean()
    z = rng.normal(0, 3, size=10)
    z -= z.mean()
    print("Teacher soft targets at three temperatures (10 classes, sorted):")
    for T in (1, 4, 20):
        print(f"  T={T:>2}: " + " ".join(f"{x:.3f}" for x in np.sort(softmax(v, T))[::-1]))
    print("\nSize of the soft-loss gradient with respect to the student's logits")
    print(f"{'T':>5} {'|grad|':>10} {'T^2 |grad|':>11} {'|(z-v)/N|':>10}")
    lm = np.linalg.norm((z - v) / len(v))
    for T in (1, 2, 4, 8, 20, 100):
        g = np.linalg.norm(soft_ce_grad(z, v, T))
        print(f"{T:5d} {g:10.5f} {T * T * g:11.5f} {lm:10.5f}")
