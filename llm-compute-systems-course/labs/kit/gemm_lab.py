"""L4 CPU GEMM shape study. Python K-tiling is pedagogical, not an optimized kernel."""
import argparse
import json
import platform
import torch
from measurement import benchmark


def tiled_gemm(a,b,tile_k=32):
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('incompatible matrix shapes')
    if a.dtype!=b.dtype or a.device!=b.device:raise ValueError('dtype/device mismatch')
    if isinstance(tile_k,bool) or not isinstance(tile_k,int) or tile_k<1:raise ValueError('tile_k must be positive integer')
    out=torch.zeros(a.shape[0],b.shape[1],dtype=a.dtype,device=a.device)
    for start in range(0,a.shape[1],tile_k):out.add_(a[:,start:start+tile_k]@b[start:start+tile_k,:])
    return out


def grouped_gemm(inputs,weights,tile_k=None):
    if len(inputs)!=len(weights):raise ValueError("expert input/weight counts differ")
    return [a@b if tile_k is None else tiled_gemm(a,b,tile_k) for a,b in zip(inputs,weights)]


def run(repeats=5):
    torch.manual_seed(11);torch.set_num_threads(1)
    rows=[]
    for name,m,n,k in [('prefill-like',128,256,256),('decode-like',1,256,256),('expert-small',7,128,131)]:
        a=torch.randn(m,k);b=torch.randn(k,n)
        reference=(a.double()@b.double()).float()
        variants={}
        for variant,fn in [('framework-mm',lambda:a@b),('python-k-tiled',lambda:tiled_gemm(a,b))]:
            actual=fn();torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-4)
            variants[variant]={'maximum_error':(actual-reference).abs().max().item(),
                'timing':benchmark(lambda:fn().sum().item(),repeats=repeats,warmup=2)}
        rows.append({'name':name,'shape_MNK':[m,n,k],'flops':2*m*n*k,'variants':variants})
    expert_lengths=[0,3,7,19]
    inputs=[torch.randn(m,131) for m in expert_lengths]
    weights=[torch.randn(131,128) for _ in expert_lengths]
    reference=grouped_gemm(inputs,weights)
    grouped={}
    for name,tile in [('framework-per-expert',None),('python-k-tiled-per-expert',32)]:
        actual=grouped_gemm(inputs,weights,tile)
        for a,b in zip(actual,reference):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
        grouped[name]=benchmark(lambda:sum(v.sum().item() for v in grouped_gemm(inputs,weights,tile)),repeats=repeats,warmup=2)
    return {'schema_version':1,'lab':'L4' ,'provenance':'measured CPU wall time; checksum and Python dispatch included',
        'environment':{'torch':torch.__version__,'python':platform.python_version(),'threads':torch.get_num_threads(),'device':'cpu'},
        'caveat':'K-tiling loop is a correctness/traffic teaching model, not a CUDA Tensor Core implementation',
        'results':rows,'grouped_cpu_exercise':{'tokens_per_expert':expert_lengths,'variants':grouped,'caveat':'Python dispatch loop, not a fused grouped GPU kernel'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    print(json.dumps(run(args.repeats),indent=2,allow_nan=False))
