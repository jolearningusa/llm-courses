"""Ratings from pairwise votes: Bradley-Terry by maximum likelihood, against online Elo (deck 32).

Under Bradley-Terry, model i beats model j with probability sigmoid(s_i - s_j). Chatbot Arena fits the
strengths s by maximum likelihood on all votes at once (a logistic regression with one +1 and one -1
feature per vote) instead of updating Elo scores one vote at a time, because the fit does not depend on the
order of the votes and gives confidence intervals. Scores are reported on the Elo scale:
400 / ln(10) * s + 1000.
"""
import numpy as np

SCALE = 400 / np.log(10)


def simulate_votes(strengths, n_votes, rng):
    """Random pairs of distinct models; the winner is drawn from the Bradley-Terry probability."""
    m = len(strengths)
    a = rng.integers(0, m, n_votes)
    b = (a + rng.integers(1, m, n_votes)) % m
    p_a = 1 / (1 + np.exp(-(strengths[a] - strengths[b])))
    a_wins = rng.random(n_votes) < p_a
    return np.where(a_wins, a, b), np.where(a_wins, b, a)


def fit_bt(winners, losers, m, iters=50, ridge=1e-6):
    """Newton's method on the Bradley-Terry log-likelihood; strengths centered at zero."""
    s = np.zeros(m)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(s[winners] - s[losers])))
        g = np.zeros(m)
        np.add.at(g, winners, 1 - p)
        np.add.at(g, losers, -(1 - p))
        w = p * (1 - p)
        H = np.zeros((m, m))
        np.add.at(H, (winners, winners), w)
        np.add.at(H, (losers, losers), w)
        np.add.at(H, (winners, losers), -w)
        np.add.at(H, (losers, winners), -w)
        s += np.linalg.solve(H + ridge * np.eye(m) + 1.0 / m, g)
        s -= s.mean()
    return s


def elo(winners, losers, m, k=4.0):
    """Online Elo on the same votes, in order; returns natural-log strengths for comparison."""
    r = np.full(m, 1000.0)
    for w, l in zip(winners, losers):
        e = 1 / (1 + 10 ** ((r[l] - r[w]) / 400))
        r[w] += k * (1 - e)
        r[l] -= k * (1 - e)
    s = (r - 1000) / SCALE
    return s - s.mean()


def bootstrap(winners, losers, m, n_boot, rng):
    """Percentile intervals from refitting on votes resampled with replacement."""
    n = len(winners)
    fits = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        fits.append(fit_bt(winners[i], losers[i], m))
    return np.percentile(np.array(fits), [2.5, 97.5], axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true = np.array([1.2, 0.8, 0.5, 0.0, -0.4, -0.9, -1.2])
    true -= true.mean()
    m = len(true)
    w, l = simulate_votes(true, 5000, rng)
    bt = fit_bt(w, l, m)
    lo, hi = bootstrap(w, l, m, 200, rng)
    order = rng.permutation(len(w))
    e1, e2 = elo(w, l, m), elo(w[order], l[order], m)
    bt2 = fit_bt(w[order], l[order], m)
    print("7 models, 5,000 votes; ratings on the Elo scale (1000 = average)")
    print(f"{'model':>5} {'true':>7} {'BT':>7} {'95% CI':>15} {'Elo':>7} {'Elo, shuffled':>14} {'BT, shuffled':>13}")
    for i in range(m):
        print(f"{i:5d} {1000 + SCALE * true[i]:7.0f} {1000 + SCALE * bt[i]:7.0f} "
              f"{1000 + SCALE * lo[i]:7.0f}-{1000 + SCALE * hi[i]:<7.0f} {1000 + SCALE * e1[i]:7.0f} "
              f"{1000 + SCALE * e2[i]:14.0f} {1000 + SCALE * bt2[i]:13.0f}")
