"""Categorical sampling reference and analytical idle-state policy; no model calls."""
import json
import math
import random


def distributions(p,q):
    if len(p)!=len(q) or not p:raise ValueError('matching nonempty categorical distributions required')
    for v in (p,q):
        if any(not math.isfinite(x) or x<0 for x in v) or not math.isclose(sum(v),1.,abs_tol=1e-10):raise ValueError('finite normalized probabilities required')
    residual=[max(a-b,0.) for a,b in zip(p,q)]
    z=sum(residual)
    return [v/z for v in residual] if z else None


def marginal(p,q):
    residual=distributions(p,q)
    rejected=1-sum(min(a,b) for a,b in zip(p,q))
    return [min(a,b)+rejected*(residual[i] if residual else 0.) for i,(a,b) in enumerate(zip(p,q))]


def choose(prob,rng):
    u=rng.random();c=0.
    for i,v in enumerate(prob):
        c+=v
        if u<c:return i
    return len(prob)-1


def sample(p,q,rng):
    residual=distributions(p,q)
    token=choose(q,rng)
    if rng.random()<min(1.,p[token]/q[token]):return token,True
    if residual is None:raise RuntimeError('rejection impossible for equal distributions')
    return choose(residual,rng),False


def expected_tokens(acceptance,draft_length):
    if not 0<=acceptance<=1 or not isinstance(draft_length,int) or draft_length<1:raise ValueError('invalid independent-acceptance assumptions')
    return sum(acceptance**i for i in range(draft_length+1))


def wait_policy(policy,state_bytes,wait_s,recompute_s,transfer_out_s=0.,transfer_in_s=0.,ttl_s=0.,identity_valid=True):
    values=(state_bytes,wait_s,recompute_s,transfer_out_s,transfer_in_s,ttl_s)
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError('finite nonnegative costs required')
    transferred=0
    if policy=='retain':
        idle=wait_s*state_bytes;resume=0. if identity_valid else recompute_s
    elif policy=='evict':idle=0.;resume=recompute_s
    elif policy=='ttl':
        idle=min(wait_s,ttl_s)*state_bytes
        resume=0. if wait_s<=ttl_s and identity_valid else recompute_s
    elif policy=='offload':
        idle=min(wait_s,transfer_out_s)*state_bytes
        transferred=state_bytes*(2 if identity_valid else 1)
        resume=max(0.,transfer_out_s-wait_s)+(transfer_in_s if identity_valid else recompute_s)
    else:raise ValueError('unknown policy')
    return {'policy':policy,'wait_s':wait_s,'identity_valid_at_resume':identity_valid,'idle_hbm_byte_seconds':idle,'extra_resume_seconds':resume,'transferred_bytes':transferred,'provenance':'analytical single-wait accounting; no contention/capacity/model execution','scope':'byte-seconds only within tool wait; outbound copy may overlap wait; inbound allocation/recompute residency excluded; identity checked at resume'}


def main():
    p,q=[0.1,0.3,0.6],[0.5,0.4,0.1]
    rng=random.Random(42);counts=[0,0,0];accepted=0
    for _ in range(20000):
        token,ok=sample(p,q,rng);counts[token]+=1;accepted+=ok
    print(json.dumps({'lab':'L10','provenance':'categorical probability experiment and analytical cache-policy model, not inference benchmark','target':p,'draft':q,'exact_marginal':marginal(p,q),'sample_frequencies':[n/20000 for n in counts],'acceptance':accepted/20000,'expected_tokens_k4_alpha08':expected_tokens(.8,4),'wait_cases':[wait_policy(policy,512*1024**2,wait,.03,.01,.01,.1) for wait in (.02,2.) for policy in ('retain','evict','ttl','offload')]},indent=2))

if __name__=='__main__':main()
