"""Explicitly simplified analytical layout accounting + CPU recomputation check."""
import argparse
import json


def plan(ranks, dp, tp, pp, cp, parameters, state_bytes_per_parameter, activation_bytes, workspace_bytes, microbatches, local_batch):
    integers=(ranks,dp,tp,pp,cp,parameters,microbatches,local_batch)
    if any(not isinstance(x,int) or x<1 for x in integers) or dp*tp*pp*cp!=ranks:
        raise ValueError('positive integer dimensions must multiply to ranks')
    if state_bytes_per_parameter<=0 or activation_bytes<0 or workspace_bytes<0:
        raise ValueError('invalid byte assumptions')
    # Data ranks fully shard persistent state; context ranks replicate parameters.
    persistent=parameters*state_bytes_per_parameter/(dp*tp*pp)
    return {'provenance':'analytical, ideal even full data sharding; context replicates state', 'persistent_bytes_per_rank':persistent,'estimated_peak_bytes_per_rank':persistent+activation_bytes+workspace_bytes,'ideal_fill_drain_utilization':microbatches/(microbatches+pp-1),'global_batch':dp*microbatches*local_batch}


def checkpoint_check():
    import copy
    import torch
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(11)
    block=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,8))
    other=copy.deepcopy(block)
    x=torch.randn(4,8,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    a=block(x)
    b=checkpoint(other,y,use_reentrant=False)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(x.grad,y.grad)
    for p,q in zip(block.parameters(),other.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    return {'provenance':'CPU output and gradient correctness, not GPU memory measurement','torch':torch.__version__,'passed':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',action='store_true')
    args=parser.parse_args()
    result={'lab':'L8','layouts':[plan(8,2,2,2,1,1_000_000_000,16,2e9,1e9,8,2),plan(8,2,1,4,1,1_000_000_000,16,2e9,1e9,8,2)]}
    if args.checkpoint:result['recomputation']=checkpoint_check()
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
