"""Frozen three-arm toy RL steering experiment; not an ASE paper reproduction."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np
import torch

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
from policy import Policy
from grpo import sft, group_advantages
from task import EOS, MAX_LEN, VOCAB, reward
PROTOCOL = LAB / 'steering_protocol.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode());h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


class SteeringPolicy(Policy):
    direction = None
    def logits(self,a,b,prev,pos):
        hidden=self.mlp[1](self.mlp[0](self.a(a)+self.b(b)+self.prev(prev)+self.pos(pos)))
        if self.direction is not None:
            hidden=hidden+(self.direction[:,None,:] if hidden.ndim==3 else self.direction)
        return self.mlp[2](hidden)

    @torch.no_grad()
    def readout(self,a,b,tokens):
        n,t=tokens.shape
        prev=torch.cat([torch.full((n,1),len(VOCAB),dtype=torch.long),tokens[:,:-1]],1)
        pos=torch.arange(t).expand(n,t)
        h=self.mlp[1](self.mlp[0](self.a(a[:,None])+self.b(b[:,None])+self.prev(prev)+self.pos(pos)))
        _,mask=self.logprobs(a,b,tokens)
        index=torch.floor(.25*(mask.sum(1)-1)).long()
        return h[torch.arange(n),index].clone(),int(mask.sum())


@contextmanager
def steering(policy,direction):
    previous=policy.direction;policy.direction=direction
    try: yield
    finally: policy.direction=previous


def pair_indices(rewards,g,seed):
    rng=random.Random(seed);pairs=[]
    for start in range(0,len(rewards),g):
        positive=[i for i in range(start,start+g) if rewards[i]==1]
        negative=[i for i in range(start,start+g) if rewards[i]==0]
        rng.shuffle(positive);rng.shuffle(negative)
        pairs.extend(zip(positive[:min(len(positive),len(negative))],negative))
    return pairs


def fit_candidates(buffer,config):
    positive=torch.stack([x[0] for x in buffer]);negative=torch.stack([x[1] for x in buffer])
    pooled=torch.cat([positive,negative]);pooled=pooled-pooled.mean(0)
    _,_,vh=torch.linalg.svd(pooled,full_matrices=False)
    projected=(positive-negative)@vh[:min(config['pca_dim'],len(vh))].T
    first=projected[0];second=projected[((projected-first)**2).sum(1).argmax()]
    centers=torch.stack([first,second])
    for _ in range(config['kmeans_iterations']):
        labels=((projected[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)
        if any(not bool((labels==k).any()) for k in range(config['K'])): return None
        centers=torch.stack([projected[labels==k].mean(0) for k in range(config['K'])])
    raw=torch.stack([(positive-negative)[labels==k].mean(0) for k in range(config['K'])])
    if not bool(torch.isfinite(raw).all()) or bool((raw.norm(dim=1)<1e-10).any()): return None
    return raw


def smooth_bank(old,candidates,beta):
    if old is None:return candidates.clone()
    a=old/old.norm(dim=1,keepdim=True);b=candidates/candidates.norm(dim=1,keepdim=True)
    direct=(a*b).sum();swapped=(a*b.flip(0)).sum()
    aligned=candidates.flip(0) if swapped>direct else candidates
    return beta*old+(1-beta)*aligned


def corrected_loss(new,old,behavior,advantages,mask,config):
    importance=torch.exp(old-behavior).clamp(max=config['importance_cap']).detach()
    ratio=torch.exp(new-old)
    a=advantages[:,None]
    clipped=torch.minimum(ratio*a,ratio.clamp(1-config['clip'],1+config['clip'])*a)
    return -(clipped*importance*mask).sum()/mask.sum(),importance


def routes(bank,n,config):
    if bank is None:return None
    ids=torch.arange(n)%config['responses_per_prompt']%3
    directions=torch.zeros(n,bank.shape[1])
    for k in range(config['K']):directions[ids==k+1]=bank[k]/bank[k].norm()*config['alpha']
    return directions


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def identity(config):
    import importlib.metadata as metadata
    for name in ('torch','numpy'):
        if metadata.version(name)!=config['runtime'][name]:raise RuntimeError('Unqualified '+name)
    return {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
            'protocol_sha256':digest(PROTOCOL),'runner_sha256':digest(Path(__file__)),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=LAB,text=True).strip(),
            'source_sha256':{p.name:digest(p) for p in [LAB/'policy.py',LAB/'grpo.py',LAB/'task.py']}}


def train_run(config,condition,seed,directory):
    start=time.perf_counter();torch.manual_seed(seed);torch.set_num_threads(1)
    record=identity(config);policy=SteeringPolicy(hidden=config['hidden'])
    tick=time.perf_counter();sft(policy,[tuple(x) for x in config['sft_prompts']],steps=config['sft_steps'],lr=config['sft_lr'],seed=seed)
    record.update(seed=seed,condition=condition,base_checkpoint_sha256=state_hash(policy.state_dict()),sft_seconds=time.perf_counter()-tick)
    opt=torch.optim.Adam(policy.parameters(),lr=config['learning_rate'])
    rng=random.Random(seed+1000);sampling=torch.Generator().manual_seed(seed+2000)
    train=[tuple(x) for x in config['train_prompts']];bank=None;buffer=[];last_update=-100;peak_storage=0;trace=[]
    totals={name:0 for name in ['sampling_seconds','scoring_seconds','replay_seconds','bank_seconds','update_seconds','sampled_slots','valid_tokens','replayed_trajectories','replayed_valid_tokens','bank_updates','bank_fit_attempts','steered_responses','discarded_buffer_pairs']}
    for step in range(config['rl_steps']):
        batch=[rng.choice(train) for _ in range(config['prompts_per_step'])]
        a=torch.tensor([x[0] for x in batch]).repeat_interleave(config['responses_per_prompt']);b=torch.tensor([x[1] for x in batch]).repeat_interleave(config['responses_per_prompt'])
        directions=routes(bank,len(a),config)
        tick=time.perf_counter()
        with steering(policy,directions):tokens=policy.sample(a,b,generator=sampling)
        totals['sampling_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();r=torch.tensor([reward((int(x),int(y)),ts.tolist()) for x,y,ts in zip(a,b,tokens)])
        with torch.no_grad():
            with steering(policy,directions):behavior,mask=policy.logprobs(a,b,tokens)
            old,_=policy.logprobs(a,b,tokens)
        advantage=group_advantages(r.view(-1,config['responses_per_prompt'])).view(-1)
        totals['scoring_seconds']+=time.perf_counter()-tick
        pairs=pair_indices(r.tolist(),config['responses_per_prompt'],seed*100000+step)
        staged=None;attempt=False;replayed=0
        if condition!='none' and pairs:
            tick=time.perf_counter();indices=[i for pair in pairs for i in pair]
            h,valid=policy.readout(a[indices],b[indices],tokens[indices])
            buffer.extend((h[i].clone(),h[i+1].clone()) for i in range(0,len(h),2))
            replayed=len(indices);totals['replayed_trajectories']+=replayed;totals['replayed_valid_tokens']+=valid
            totals['replay_seconds']+=time.perf_counter()-tick
            if len(buffer)>config['buffer_capacity']:
                totals['discarded_buffer_pairs']+=len(buffer)-config['buffer_capacity'];buffer=buffer[-config['buffer_capacity']:]
        storage=sum(x.numel()*x.element_size()+y.numel()*y.element_size() for x,y in buffer)+(bank.numel()*bank.element_size() if bank is not None else 0)
        peak_storage=max(peak_storage,storage)
        if condition!='none' and len(buffer)>=config['minimum_pairs'] and step-last_update>=config['bank_interval']:
            tick=time.perf_counter();attempt=True;totals['bank_fit_attempts']+=1;candidates=fit_candidates(buffer,config)
            if candidates is not None:
                if condition=='random':
                    random_directions=torch.randn(candidates.shape,generator=torch.Generator().manual_seed(seed*10000+step+9000))
                    candidates=random_directions/random_directions.norm(dim=1,keepdim=True)*candidates.norm(dim=1,keepdim=True)
                staged=smooth_bank(bank,candidates,config['ema_beta'])
                if bool((staged.norm(dim=1)<1e-10).any()):staged=None
            totals['bank_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter();losses=[];weight_range=None
        for _ in range(config['inner_updates']):
            new,_=policy.logprobs(a,b,tokens)
            loss,weights=corrected_loss(new,old,behavior,advantage,mask,config)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite update')
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            weight_range=[float(weights[mask.bool()].min()),float(weights[mask.bool()].max())]
        totals['update_seconds']+=time.perf_counter()-tick
        active=0 if directions is None else int((directions.norm(dim=1)>0).sum())
        if staged is not None:
            bank=staged.detach();buffer=[];last_update=step;totals['bank_updates']+=1
        totals['steered_responses']+=active;totals['sampled_slots']+=tokens.numel();totals['valid_tokens']+=int(mask.sum())
        grouped=r.view(-1,config['responses_per_prompt'])
        trace.append({'step':step,'sampled_slots':tokens.numel(),'valid_tokens':int(mask.sum()),'positive_rewards':int(r.sum()),
                      'mixed_groups':int((grouped.min(1).values!=grouped.max(1).values).sum()),
                      'paired_trajectory_indices':pairs if condition!='none' else [],'replayed_trajectories':replayed,
                      'bank_fit_attempted':attempt,'bank_published_after_update':staged is not None,
                      'bank_sha256':state_hash({'bank':bank}) if bank is not None else None,
                      'rollout_steered_responses':active,'importance_weight_range':weight_range,'losses':losses,
                      'rollout_tokens_sha256':hashlib.sha256(tokens.numpy().tobytes()).hexdigest()})
    assert policy.direction is None
    final=directory/'final.pt';torch.save(policy.state_dict(),final)
    record.update(trace=trace,totals=totals,checkpoint_file_sha256=digest(final),final_parameter_sha256=state_hash(policy.state_dict()),
                  checkpoint='final.pt',bank_active=bank is not None,final_bank=bank.tolist() if bank is not None else [],bank_tensor_bytes=bank.numel()*bank.element_size() if bank is not None else 0,
                  peak_bank_buffer_tensor_bytes=peak_storage,whole_worker_peak_rss_bytes=rss_bytes(),
                  phase_wall_seconds=time.perf_counter()-start,selection='fixed final step; heldout not read/evaluated')
    return record


def evaluate_run(config,directory):
    start=time.perf_counter();torch.set_num_threads(1);record=identity(config)
    training=json.loads((directory/'train.json').read_text())
    if training['protocol_sha256']!=digest(PROTOCOL) or training['runner_sha256']!=digest(Path(__file__)):raise ValueError('Changed training contract')
    checkpoint=directory/'final.pt'
    if digest(checkpoint)!=training['checkpoint_file_sha256']:raise ValueError('Changed checkpoint')
    policy=SteeringPolicy(hidden=config['hidden']);policy.load_state_dict(torch.load(checkpoint,weights_only=True));policy.eval()
    assert policy.direction is None
    ps=config['heldout_prompts'];a=torch.tensor([p[0] for p in ps]).repeat_interleave(config['heldout_samples']);b=torch.tensor([p[1] for p in ps]).repeat_interleave(config['heldout_samples'])
    tick=time.perf_counter()
    tokens=policy.sample(a,b,generator=torch.Generator().manual_seed(config['heldout_seed']))
    elapsed=time.perf_counter()-tick
    scores=[reward((int(x),int(y)),t.tolist()) for x,y,t in zip(a,b,tokens)]
    rows=[]
    for i,prompt in enumerate(ps):
        lo=i*config['heldout_samples'];hi=lo+config['heldout_samples'];r=scores[lo:hi]
        rows.append({'prompt':prompt,'responses':tokens[lo:hi].tolist(),'rewards':r,'mean_reward':sum(r)/len(r),'pass_at_8':any(r)})
    record.update(seed=training['seed'],condition=training['condition'],rows=rows,steering_disabled=True,
                  checkpoint_sha256=digest(checkpoint),parameter_sha256=state_hash(policy.state_dict()),
                  mean_reward=sum(scores)/len(scores),pass_at_8=sum(r['pass_at_8'] for r in rows)/len(rows),
                  sampled_slots=tokens.numel(),sample_seconds=elapsed,whole_worker_peak_rss_bytes=rss_bytes(),phase_wall_seconds=time.perf_counter()-start)
    assert record['parameter_sha256']==training['final_parameter_sha256']
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['train','evaluate']);parser.add_argument('--condition',choices=['none','random','learned']);parser.add_argument('--seed',type=int);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();config=json.loads(PROTOCOL.read_text())
    if args.phase=='train':
        if args.condition is None or args.seed not in config['seeds']:parser.error('Frozen condition/seed required')
        if args.directory.exists():raise FileExistsError('Use a new training directory')
        args.directory.mkdir(parents=True);record=train_run(config,args.condition,args.seed,args.directory);name='train.json'
    else:
        if (args.directory/'evaluation.json').exists():raise FileExistsError('Preserve prior evaluation')
        record=evaluate_run(config,args.directory);name='evaluation.json'
    (args.directory/name).write_text(json.dumps(record,indent=2)+'\n')
    print(args.phase,record['condition'],record['seed'],'recorded',flush=True)
