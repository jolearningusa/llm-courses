"""L2: small CPU eager/compiled forward and backward with explicit timing scope."""
import argparse
import copy
import json
import platform
import time
import torch
from measurement import benchmark


def block(width=32):
    return torch.nn.Sequential(torch.nn.Linear(width,width),torch.nn.LayerNorm(width),torch.nn.GELU())


def execute(model,x,training):
    model.zero_grad(set_to_none=True)
    x.grad=None
    if training:
        y=model(x);y.square().mean().backward()
    else:
        with torch.no_grad():y=model(x)
    return y


def compare(reference,candidate,x,training,rtol=1e-4,atol=1e-5):
    left=x.detach().clone().requires_grad_(training)
    right=x.detach().clone().requires_grad_(training)
    a=execute(reference,left,training).detach().clone()
    b=execute(candidate,right,training).detach().clone()
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    if training:
        torch.testing.assert_close(left.grad,right.grad,rtol=rtol,atol=atol)
        for p,q in zip(reference.parameters(),candidate.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=rtol,atol=atol)
    return {'passed':True,'maximum_output_error':(a-b).abs().max().item(),
            'gradient_check':training,'rtol':rtol,'atol':atol}


def saved_state(model,x):
    records=[]
    def pack(t):
        records.append({'shape':list(t.shape),'dtype':str(t.dtype),'logical_bytes':t.numel()*t.element_size()})
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):execute(model,x,True)
    return {'records':records,'summed_logical_bytes':sum(r['logical_bytes'] for r in records),
            'caveat':'references may alias; this sum is not unique storage or peak allocation'}


def run(compiled=False,training=False,dynamic=False,repeats=5):
    torch.manual_seed(7);torch.set_num_threads(1)
    eager=block();candidate=copy.deepcopy(eager)
    if compiled:candidate=torch.compile(candidate,backend='inductor',dynamic=dynamic,fullgraph=True)
    results=[]
    for batch in (4,7):
        x=torch.randn(batch,32,requires_grad=training)
        cold={}
        for name,model in [('eager',eager),('candidate',candidate)]:
            start=time.perf_counter();execute(model,x,training);cold[name]=time.perf_counter()-start
        correctness=compare(eager,candidate,x,training)
        timing={}
        for name,model in [('eager',eager),('candidate',candidate)]:
            def work(model=model):
                y=execute(model,x,training)
                # The checksum consumes the completed CPU result; identical scope for both.
                return y.detach().sum().item()
            timing[name]=benchmark(work,repeats=repeats,warmup=2)
        results.append({'batch':batch,'first_use_wall_seconds':cold,'correctness':correctness,'warm':timing})
    probe=torch.randn(4,32,requires_grad=True)
    return {'schema_version':1,'lab':'L2','provenance':'measured CPU wall time',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform(),'threads':torch.get_num_threads()},
            'backend':'inductor' if compiled else 'eager-copy-control',
            'dynamic':dynamic,'training':training,
            'scope':'forward + gradient clearing + checksum' + (' + backward; excludes optimizer' if training else '; no gradients'),
            'results':results,'fx_graph':str(torch.fx.symbolic_trace(eager).graph),
            'eager_saved_state':saved_state(eager,probe)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--training',action='store_true')
    parser.add_argument('--dynamic',action='store_true')
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    print(json.dumps(run(args.compile,args.training,args.dynamic,args.repeats),indent=2,allow_nan=False))
