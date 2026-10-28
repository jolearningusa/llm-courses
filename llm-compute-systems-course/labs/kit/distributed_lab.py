"""Two-process CPU Gloo DDP correctness experiment, not a GPU benchmark."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import time


def ring_seconds(ranks, payload_bytes, latency_s, bandwidth_bytes_s):
    if ranks < 1 or payload_bytes < 0 or latency_s < 0 or bandwidth_bytes_s <= 0:
        raise ValueError('invalid ring assumptions')
    return 2 * (ranks-1) * latency_s + 2 * (ranks-1) * payload_bytes / (ranks * bandwidth_bytes_s)


def worker(rank, world, store, output, delay):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + store, rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(23)
        model = torch.nn.Linear(8,4)
        reference = copy.deepcopy(model)
        ddp = DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.03)
        refopt = torch.optim.SGD(reference.parameters(), lr=0.03)
        samples=[]
        max_error=0.
        for step in range(3):
            # Generate the identical global batch on each rank, then disjoint equal shards.
            torch.manual_seed(100+step)
            x,y=torch.randn(32,8),torch.randn(32,4)
            start,end=rank*16,(rank+1)*16
            dist.barrier()
            tick=time.perf_counter()
            if rank==1:
                time.sleep(delay)
            opt.zero_grad(set_to_none=True)
            loss=(ddp(x[start:end])-y[start:end]).square().mean()
            loss.backward()
            opt.step()
            elapsed=torch.tensor(time.perf_counter()-tick,dtype=torch.float64)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            samples.append(elapsed.item())
            refopt.zero_grad(set_to_none=True)
            (reference(x)-y).square().mean().backward()
            refopt.step()
            for actual,expected in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                max_error=max(max_error,(actual-expected).abs().max().item())
        if rank==0:
            Path(output).write_text(json.dumps({'lab':'L12','provenance':'measured local CPU process experiment','torch':torch.__version__,'backend':'gloo','ranks':world,'global_batch':32,'steps':3,'rank1_delay_s':delay,'max_rank_step_wall_seconds':samples,'parameter_max_abs_error':max_error,'timing_scope':'local forward/backward/update plus injected delay; max reduction after timer; barrier outside'},indent=2))
    finally:
        dist.destroy_process_group()


def main():
    import torch.multiprocessing as mp
    parser=argparse.ArgumentParser()
    parser.add_argument('--rank-delay',type=float,default=0.)
    args=parser.parse_args()
    if args.rank_delay<0:
        parser.error('delay must be nonnegative')
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,args=(2,os.path.join(tmp,'store'),os.path.join(tmp,'result.json'),args.rank_delay),nprocs=2,join=True)
        print(Path(tmp,'result.json').read_text())

if __name__=='__main__':
    main()
