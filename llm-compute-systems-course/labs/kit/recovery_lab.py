"""Trusted local CPU checkpoint/resume reference, including dropout RNG and cursor."""
import json
import os
from pathlib import Path
import tempfile


def make():
    import torch
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ReLU(),torch.nn.Dropout(0.3),torch.nn.Linear(8,2))
    return model,torch.optim.AdamW(model.parameters(),lr=0.01)


def train(model,opt,start,stop,data):
    losses=[]
    for cursor in range(start,stop):
        x,y=data[cursor]
        opt.zero_grad(set_to_none=True)
        loss=(model(x)-y).square().mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def experiment(restore_rng=True):
    import torch
    torch.set_num_threads(1)
    generator=torch.Generator().manual_seed(77)
    data=[(torch.randn(8,4,generator=generator),torch.randn(8,2,generator=generator)) for _ in range(6)]
    torch.manual_seed(31)
    baseline,baseopt=make()
    baseline_losses=train(baseline,baseopt,0,6,data)
    torch.manual_seed(31)
    model,opt=make()
    first=train(model,opt,0,3,data)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory,'checkpoint.pt')
        # Rename protects readers from a partly-written local file; not a durability guarantee.
        pending=path.with_suffix('.pending')
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cursor':3,'format':1},pending)
        os.replace(pending,path)
        checkpoint_bytes=path.stat().st_size
        # Fresh construction consumes RNG, as a real restart often does.
        resumed,resopt=make()
        state=torch.load(path,weights_only=True)
        if state['format']!=1:raise ValueError('unsupported checkpoint format')
        resumed.load_state_dict(state['model'])
        resopt.load_state_dict(state['optimizer'])
        if restore_rng:torch.set_rng_state(state['rng'])
        remaining=train(resumed,resopt,state['cursor'],6,data)
    error=max((a-b).abs().max().item() for a,b in zip(baseline.parameters(),resumed.parameters()))
    optimizer_error=max((baseopt.state[a][key]-resopt.state[b][key]).abs().max().item() for a,b in zip(baseline.parameters(),resumed.parameters()) for key in ('step','exp_avg','exp_avg_sq'))
    return {'lab':'L14','provenance':'actual local CPU save/load and deterministic synthetic replay','torch':torch.__version__,'checkpoint_bytes':checkpoint_bytes,'restored_rng':restore_rng,'baseline_losses':baseline_losses,'resumed_losses':first+remaining,'parameter_max_abs_error':error,'optimizer_max_abs_error':optimizer_error,'data_cursor':6}

if __name__=='__main__':
    result=experiment()
    if result['parameter_max_abs_error']!=0:raise RuntimeError('resume diverged')
    print(json.dumps(result,indent=2))
