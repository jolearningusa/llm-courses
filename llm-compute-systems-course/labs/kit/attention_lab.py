"""L5 CPU single-head square self-attention references; not fused GPU kernels."""
import argparse
import json
import math
import torch
from measurement import benchmark


def validate(q,k,v):
    if any(t.ndim!=2 for t in (q,k,v)):raise ValueError('single-head matrices required')
    if q.shape[0]!=k.shape[0] or k.shape[0]!=v.shape[0] or q.shape[1]!=k.shape[1]:raise ValueError('square self-attention shape mismatch')
    if min(q.shape[0],q.shape[1],v.shape[1])<1:raise ValueError('nonempty dimensions required')
    if any(t.device.type!='cpu' for t in (q,k,v)):raise ValueError('this teaching reference is CPU-only')
    if q.dtype not in (torch.float32,torch.float64) or any(t.dtype!=q.dtype for t in (k,v)):raise ValueError('matching float32/64 required')


def full_attention(q,k,v,causal=False):
    validate(q,k,v)
    scores=q@k.T/math.sqrt(q.shape[1])
    if causal:scores=scores.masked_fill(torch.ones(q.shape[0],k.shape[0],dtype=torch.bool).triu(1),float('-inf'))
    return torch.softmax(scores,dim=-1)@v


def tiled_attention(q,k,v,causal=False,tile_q=4,tile_k=5):
    validate(q,k,v)
    for tile in (tile_q,tile_k):
        if isinstance(tile,bool) or not isinstance(tile,int) or tile<1:raise ValueError('positive integer tiles required')
    outputs=[];length=q.shape[0]
    for qs in range(0,length,tile_q):
        query=q[qs:qs+tile_q];rows=query.shape[0]
        maximum=torch.full((rows,1),float('-inf'),dtype=q.dtype)
        normalizer=torch.zeros(rows,1,dtype=q.dtype)
        weighted=torch.zeros(rows,v.shape[1],dtype=q.dtype)
        for ks in range(0,length,tile_k):
            key=k[ks:ks+tile_k];value=v[ks:ks+tile_k]
            scores=query@key.T/math.sqrt(q.shape[1])
            if causal:
                mask=torch.arange(ks,ks+key.shape[0])[None,:]>torch.arange(qs,qs+rows)[:,None]
                scores=scores.masked_fill(mask,float('-inf'))
            updated=torch.maximum(maximum,scores.max(dim=-1,keepdim=True).values)
            old_scale=torch.exp(maximum-updated)
            probabilities=torch.exp(scores-updated)
            normalizer=normalizer*old_scale+probabilities.sum(dim=-1,keepdim=True)
            weighted=weighted*old_scale+probabilities@value
            maximum=updated
        outputs.append(weighted/normalizer)
    return torch.cat(outputs,dim=0)


def run(length=33,repeats=5):
    torch.manual_seed(17);torch.set_num_threads(1)
    q=torch.randn(length,16);k=torch.randn(length,16);v=torch.randn(length,12)
    results=[]
    for causal in (False,True):
        with torch.no_grad():
            reference=full_attention(q,k,v,causal)
            tiled=tiled_attention(q,k,v,causal)
            torch.testing.assert_close(tiled,reference,rtol=1e-4,atol=1e-5)
            timings={name:benchmark(lambda:fn(q,k,v,causal).sum().item(),repeats,2) for name,fn in [('full',full_attention),('tiled-python',tiled_attention)]}
        results.append({'causal':causal,'maximum_error':(reference-tiled).abs().max().item(),'timings':timings})
    return {'schema_version':1,'lab':'L5','provenance':'measured CPU wall time, no_grad, Python/checksum included',
            'torch':torch.__version__,'length':length,'results':results,
            'limits':'single head, CPU float32/64, equal query/key lengths; Python autograd can retain tile graphs, so this is not a fused training-memory benchmark'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--length',type=int,default=33)
    args=parser.parse_args()
    if args.length<1:parser.error('length must be positive')
    print(json.dumps(run(args.length),indent=2,allow_nan=False))
