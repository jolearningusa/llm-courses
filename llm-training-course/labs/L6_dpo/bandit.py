"""A one-prompt policy over K answers, trained by gradient descent on the expected preference loss.

The policy is a softmax over K logits. A pair (i, j) is drawn uniformly from all ordered pairs with
i != j; the label says i wins with probability P[i, j]. The expected loss is the average over pairs of
P[i, j] * loss(i beats j). Exact expectations, no sampling, so the optimum can be compared with a
closed form.
"""
import numpy as np

from losses import dpo, ipo


def log_softmax(theta):
    m = theta.max()
    return theta - m - np.log(np.exp(theta - m).sum())


def bradley_terry(r):
    """P[i, j] = sigmoid(r_i - r_j)."""
    return 1.0 / (1.0 + np.exp(-(r[:, None] - r[None, :])))


def expected_grad(theta, ref_logp, P, loss_fn, **kw):
    """Expected loss over ordered pairs and its gradient with respect to the logits."""
    K = theta.shape[0]
    logp = log_softmax(theta)
    i, j = np.where(~np.eye(K, dtype=bool))
    w = P[i, j] / len(i)
    loss, g_w, g_l = loss_fn(logp[i], logp[j], ref_logp[i], ref_logp[j], **kw)
    g = np.zeros(K)
    np.add.at(g, i, w * g_w)
    np.add.at(g, j, w * g_l)
    p = np.exp(logp)
    return (w * loss).sum(), g - p * g.sum()   # chain rule through log-softmax


def train(ref_logp, P, loss_fn, steps=5000, lr=1.0, **kw):
    theta = ref_logp.copy()
    for _ in range(steps):
        _, grad = expected_grad(theta, ref_logp, P, loss_fn, **kw)
        theta -= lr * grad
    return log_softmax(theta)


def rlhf_optimum(ref_logp, r, beta):
    """The KL-regularized RLHF optimum: pi*(y) proportional to pi_ref(y) exp(r(y) / beta)."""
    return log_softmax(ref_logp + r / beta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    K = 6
    ref = log_softmax(rng.normal(size=K))
    r = rng.normal(size=K)
    P = bradley_terry(r)
    pol = train(ref, P, dpo, lr=1 / args.beta ** 2, beta=args.beta)
    star = rlhf_optimum(ref, r, args.beta)
    print(f"DPO against the closed-form RLHF optimum, K = {K}, beta = {args.beta}")
    print(f"{'answer':>6} {'reward':>8} {'pi_ref':>8} {'pi_DPO':>8} {'pi*':>8}")
    for k in range(K):
        print(f"{k:6d} {r[k]:8.3f} {np.exp(ref[k]):8.4f} {np.exp(pol[k]):8.4f} {np.exp(star[k]):8.4f}")
    print(f"max |log pi_DPO - log pi*| = {np.abs(pol - star).max():.2e}")

    print("\nTwo answers, answer 0 always preferred, uniform reference")
    P2 = np.array([[0.0, 1.0], [0.0, 0.0]])
    ref2 = np.log(np.array([0.5, 0.5]))
    print(f"{'beta or tau':>11} {'steps':>6} {'DPO pi(1)':>10} {'IPO pi(1)':>10} {'IPO target':>10}")
    for c in (0.1, 1.0):
        for steps in (1000, 10000):
            d = np.exp(train(ref2, P2, dpo, steps=steps, beta=c))[1]
            i = np.exp(train(ref2, P2, ipo, steps=steps, lr=0.1, tau=c))[1]
            target = 1 / (1 + np.exp(1 / (2 * c)))
            print(f"{c:11.1f} {steps:6d} {d:10.2e} {i:10.4f} {target:10.4f}")
