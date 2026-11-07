"""Preference losses computed from log-probabilities, with their gradients (decks 22 and 23).

Every function takes per-pair arrays: the policy's log-probabilities of the chosen (w) and rejected (l)
answers, and where the loss needs one, the reference model's. It returns the per-pair losses and their
derivatives with respect to the policy log-probabilities, so that any model can apply the chain rule.
Nothing here needs a reward model: that is the point of these methods.
"""
import numpy as np


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def dpo(lw, ll, ref_w, ref_l, beta=0.1):
    """DPO (Rafailov et al. 2023, Eq. 7): -log sigmoid(beta * [(lw - ref_w) - (ll - ref_l)]).

    The gradient weight sigmoid(-z) is large when the implicit reward orders the pair wrongly."""
    z = beta * ((lw - ref_w) - (ll - ref_l))
    loss = softplus(-z)
    g = sigmoid(-z) * beta
    return loss, -g, g


def ipo(lw, ll, ref_w, ref_l, tau=0.1):
    """IPO (Azar et al. 2023, Eq. 17): (h - 1/(2 tau))^2, h the same log-ratio difference as DPO.

    A squared loss has a finite target, so even a pair that always goes one way stops at a finite gap."""
    h = (lw - ref_w) - (ll - ref_l)
    d = h - 1.0 / (2.0 * tau)
    return d ** 2, 2 * d, -2 * d


def simpo(lw, ll, len_w, len_l, beta=2.0, gamma=1.0):
    """SimPO (Meng et al. 2024, Eq. 6): -log sigmoid(beta * lw/|y_w| - beta * ll/|y_l| - gamma).

    The reward is the length-normalized log-likelihood, with no reference model, and the chosen answer
    must win by a margin gamma."""
    z = beta * lw / len_w - beta * ll / len_l - gamma
    loss = softplus(-z)
    s = sigmoid(-z)
    return loss, -s * beta / len_w, s * beta / len_l


def implicit_reward(logp, ref_logp, beta):
    """DPO's implicit reward beta * log(pi / pi_ref), equal to the true reward up to a per-prompt constant."""
    return beta * (logp - ref_logp)
