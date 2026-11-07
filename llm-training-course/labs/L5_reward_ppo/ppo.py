"""PPO with a KL penalty on a one-step policy over the toy responses (decks 19 and 20).

The policy tilts the reference policy along the response features,

    log pi_theta(a) = log pi_ref(a) + theta . phi(a) - log Z(theta),

and starts at theta = 0, which is the reference policy. Each iteration samples a batch, scores it with
the KL-penalized reward used in RLHF,

    R(a) = r_RM(a) - beta * (log pi(a) - log pi_ref(a)),

and takes several epochs of steps on PPO's clipped objective. With one step per episode the advantage
is R minus a baseline (the batch mean); gae() shows the general multi-step form.

The KL-regularized optimum is pi*(a) proportional to pi_ref(a) exp(r_RM(a) / beta) (deck 22 derives
it). For a linear reward model r_RM = w . phi that is theta* = w / beta, inside this family, so the lab
can check that PPO finds it.
"""
import torch


def clipped_objective(logp_new, logp_old, adv, eps=0.2):
    """PPO-clip, to be maximized: mean of min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)."""
    rho = torch.exp(logp_new - logp_old)
    return torch.min(rho * adv, torch.clamp(rho, 1 - eps, 1 + eps) * adv).mean()


def unclipped_objective(logp_new, logp_old, adv):
    """The plain importance-weighted policy-gradient surrogate: mean of rho * A."""
    return (torch.exp(logp_new - logp_old) * adv).mean()


def gae(rewards, values, gamma=1.0, lam=0.95):
    """Generalized advantage estimation for one episode.

    rewards: (T,), values: (T + 1,) with values[T] the value after the last step (0 if terminal).
    A_t = sum_l (gamma * lam)^l * delta_t+l,  with delta_t = r_t + gamma * V(t+1) - V(t).
    """
    T = rewards.shape[0]
    adv = torch.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
    return adv


def kl(logp, ref_logp):
    """KL(pi || pi_ref) for full distributions given as log-probabilities."""
    return (logp.exp() * (logp - ref_logp)).sum().item()


def policy_logp(theta, phi, ref_logp):
    return torch.log_softmax(ref_logp + phi @ theta, 0)


def ppo_step(theta, phi, ref_logp, reward, batch=512, epochs=4, lr=0.5, eps=0.2, beta=0.1,
             clip=True, generator=None):
    """One PPO iteration on theta (a leaf tensor, updated in place). Returns the batch's mean
    KL-penalized reward and KL(pi_old || pi_new), the size of the policy step."""
    with torch.no_grad():
        logp_old_all = policy_logp(theta, phi, ref_logp)
        a = torch.multinomial(logp_old_all.exp(), batch, replacement=True, generator=generator)
        logp_old = logp_old_all[a]
        R = reward[a] - beta * (logp_old - ref_logp[a])
        adv = R - R.mean()
    opt = torch.optim.SGD([theta], lr=lr)
    for _ in range(epochs):
        logp_new = policy_logp(theta, phi, ref_logp)[a]
        obj = clipped_objective(logp_new, logp_old, adv, eps) if clip else \
            unclipped_objective(logp_new, logp_old, adv)
        opt.zero_grad()
        (-obj).backward()
        opt.step()
    with torch.no_grad():
        step_kl = kl(logp_old_all, policy_logp(theta, phi, ref_logp))
    return R.mean().item(), step_kl


def train_policy(phi, ref_logp, reward, iters=300, beta=0.1, seed=0, **kw):
    """Run PPO from the reference policy. Returns theta and the final log-probabilities."""
    g = torch.Generator().manual_seed(seed)
    theta = torch.zeros(phi.shape[1], requires_grad=True)
    for _ in range(iters):
        ppo_step(theta, phi, ref_logp, reward, beta=beta, generator=g, **kw)
    theta = theta.detach()
    return theta, policy_logp(theta, phi, ref_logp)
