"""CPU rollout/learner probability contracts; no verl or distributed backend run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

from policy import Policy
from grpo import policy_loss
from task import EOS, VOCAB, reward


def fingerprint(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(bytes(value.detach().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def expected_response_mask(tokens):
    """Literal first-EOS reference independent of Policy.logprobs' cumsum."""
    rows=[]
    for row in tokens.tolist():
        end=row.index(EOS)+1 if EOS in row else len(row)
        rows.append([int(i<end) for i in range(len(row))])
    return torch.tensor(rows,dtype=torch.float32)


def valid_ratios(current, previous, mask):
    """Select valid positions BEFORE exponentiation; reject empty/invalid masks."""
    if (current.shape!=previous.shape or mask.shape!=current.shape or current.ndim!=2
            or not torch.all((mask==0)|(mask==1)) or not mask.bool().any()
            or not torch.all(mask.sum(1)>0)):
        raise ValueError('aligned per-token log-probs and nonempty binary response masks required')
    delta=(current-previous)[mask.bool()]
    if not torch.isfinite(delta).all():raise ValueError('nonfinite valid log-ratio')
    return delta.exp()


def check_same_contract(current,previous,mask,expected_mask):
    if mask.shape!=expected_mask.shape or not torch.equal(mask,expected_mask):
        raise ValueError('response mask differs from the recorded EOS convention')
    ratios=valid_ratios(current,previous,mask)
    torch.testing.assert_close(ratios,torch.ones_like(ratios),rtol=1e-12,atol=1e-12)
    return float((ratios-1).abs().max().detach())


@torch.no_grad()
def replay(policy,a,b,tokens,temperature=1.0,wrong_shift=False):
    """Independent stepwise scoring, not Policy.logprobs' batched path."""
    previous=torch.full_like(a,len(VOCAB)); result=[]
    for t in range(tokens.shape[1]):
        prev=tokens[:,t] if wrong_shift else previous
        logits=policy.logits(a,b,prev,torch.full_like(a,t))/temperature
        result.append(logits.log_softmax(-1).gather(1,tokens[:,t,None]).squeeze(1))
        previous=tokens[:,t]
    return torch.stack(result,1)


def report():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    behavior=Policy(hidden=16).double().eval()
    a,b=torch.tensor([1,2,3,4]),torch.tensor([2,3,4,5])
    captured=[]
    original=behavior.logits
    def observe(*args):
        logits=original(*args)
        captured.append(logits.detach().clone())
        return logits
    start=time.perf_counter()
    with patch.object(behavior,'logits',side_effect=observe):
        tokens=behavior.sample(a,b,generator=torch.Generator().manual_seed(29))
    rollout_seconds=time.perf_counter()-start
    start=time.perf_counter()
    rewards=[reward((int(x),int(y)),ts.tolist()) for x,y,ts in zip(a,b,tokens)]
    reward_seconds=time.perf_counter()-start
    # Actual sampling logits, before multinomial selection. T=1, no top-k/top-p.
    rollout_lp=torch.stack([x.log_softmax(-1).gather(1,tokens[:,i,None]).squeeze(1)
                            for i,x in enumerate(captured)],1)
    anchor=copy.deepcopy(behavior)
    start=time.perf_counter()
    with torch.no_grad():anchor_lp,mask=anchor.logprobs(a,b,tokens)
    rescore_seconds=time.perf_counter()-start
    expected_mask=expected_response_mask(tokens)
    error=check_same_contract(anchor_lp,rollout_lp,mask,expected_mask)
    torch.testing.assert_close(replay(behavior,a,b,tokens)[mask.bool()],
                               rollout_lp[mask.bool()],rtol=1e-12,atol=1e-12)
    current=copy.deepcopy(anchor)
    versions=dict(behavior=fingerprint(behavior),anchor=fingerprint(anchor),before=fingerprint(current))
    fresh_lp,_=current.logprobs(a,b,tokens)
    fresh_error=check_same_contract(fresh_lp,anchor_lp,mask,expected_mask)
    # Fixed signed advantages isolate the probability contract; not a reward study.
    advantages=torch.tensor([-1.,-.5,.5,1.],dtype=torch.float64)
    optimizer=torch.optim.SGD(current.parameters(),lr=.01)
    start=time.perf_counter()
    loss=policy_loss(fresh_lp,anchor_lp.detach(),advantages,mask)
    optimizer.zero_grad();loss.backward();optimizer.step()
    update_seconds=time.perf_counter()-start
    versions['after']=fingerprint(current)
    with torch.no_grad():updated_lp,_=current.logprobs(a,b,tokens)
    lag=valid_ratios(updated_lp,anchor_lp,mask)
    if versions['after']==versions['before'] or torch.allclose(lag,torch.ones_like(lag),atol=1e-12,rtol=1e-12):
        raise AssertionError('one controlled update must create observable policy lag')
    root=Path(__file__).resolve().parent
    return dict(schema=1,python=platform.python_version(),torch=str(torch.__version__),
                device='cpu',dtype='float64',threads=torch.get_num_threads(),
                config=dict(model_seed=23,sample_seed=29,hidden=16,temperature=1.,top_k=None,top_p=None),
                source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                               for n in ('rollout_consistency.py','policy.py','grpo.py','task.py')},
                versions=versions,prompts=list(zip(a.tolist(),b.tolist())),tokens=tokens.tolist(),
                response_mask=mask.tolist(),eos=EOS,eos_included=True,
                rollout_log_probs=rollout_lp.tolist(),anchor_log_probs=anchor_lp.tolist(),
                after_log_probs=updated_lp.tolist(),valid_tokens=int(mask.sum()),
                sampled_slots=tokens.numel(),actual_rewards=rewards,
                synthetic_advantages=advantages.tolist(),
                same_anchor_rollout_max_ratio_error=error,same_current_anchor_max_ratio_error=fresh_error,
                after_update_ratio_min=float(lag.min()),after_update_ratio_max=float(lag.max()),
                cpu_seconds=dict(sample=rollout_seconds,reward=reward_seconds,
                                 anchor_rescore=rescore_seconds,update=update_seconds),
                queue=dict(in_flight_batches=1,waiting_batches=0,scope='single synchronous CPU caller'),
                verl_execution='not_run',verl_queue_depth='not_measured',
                limits='No SFT/RL quality experiment, engine mismatch, GPU, Ray, distributed/async '
                       'queue or throughput measurement. CPU intervals are single-run diagnostics.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:json.dump(report(),stream,indent=2,sort_keys=True);stream.write('\n')


if __name__=='__main__':main()
