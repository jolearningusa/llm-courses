"""CPU local MoE reference; expert grouping is a Python loop, not fused GEMM."""
import json
import time


def routed(x, w1, w2, indices, gates):
    import torch
    if indices.shape!=gates.shape or indices.shape[0]!=x.shape[0]:
        raise ValueError('routing shape mismatch')
    if indices.dtype!=torch.long or (indices<0).any() or (indices>=w1.shape[0]).any():
        raise ValueError('invalid expert indices')
    if not torch.isfinite(gates).all() or (gates<0).any():
        raise ValueError('finite nonnegative gates required')
    output=torch.zeros_like(x)
    counts=[]
    for expert in range(w1.shape[0]):
        token,slot=(indices==expert).nonzero(as_tuple=True)
        counts.append(token.numel())
        if token.numel():
            local=torch.relu(x[token]@w1[expert])@w2[expert]
            output.index_add_(0,token,local*gates[token,slot,None])
    return output,counts


def token_reference(x,w1,w2,indices,gates):
    import torch
    return torch.stack([sum((torch.relu(x[t]@w1[int(indices[t,k])])@w2[int(indices[t,k])])*gates[t,k] for k in range(indices.shape[1])) for t in range(x.shape[0])])


def fixture(skew=False):
    import torch
    torch.manual_seed(9)
    x,w1,w2=torch.randn(17,8),torch.randn(4,8,16),torch.randn(4,16,8)
    indices=torch.stack([torch.arange(17)%4,(torch.arange(17)+1)%4],dim=1)
    if skew:indices[:]=torch.tensor([0,1])
    gates=torch.softmax(torch.randn(17,2),dim=-1)
    return x,w1,w2,indices,gates


def main():
    import torch
    torch.set_num_threads(1)
    results=[]
    for skew in (False,True):
        args=fixture(skew)
        tick=time.perf_counter()
        actual,counts=routed(*args)
        elapsed=time.perf_counter()-tick
        expected=token_reference(*args)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        results.append({'skew':skew,'expert_counts':counts,'max_to_mean_load':max(counts)/(sum(counts)/len(counts)),'single_cpu_wall_seconds':elapsed,'max_abs_error':(actual-expected).abs().max().item()})
    print(json.dumps({'lab':'L9','provenance':'CPU synthetic reference, single call includes Python; not a benchmark','torch':torch.__version__,'cases':results},indent=2))
if __name__=='__main__':main()
