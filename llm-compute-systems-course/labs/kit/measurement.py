"""L1 CPU timing and analytical interval accounting. No GPU benchmark is implied."""
import argparse
import json
import math
import platform
import statistics
import time


def summarize(samples):
    samples = list(samples)
    if not samples or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x < 0 for x in samples):
        raise ValueError('samples must contain finite, nonnegative timings')
    ordered = sorted(samples)
    return {'count':len(samples), 'minimum_seconds':ordered[0],
            'median_seconds':statistics.median(samples),
            'p95_nearest_rank_seconds':ordered[math.ceil(.95*len(samples))-1],
            'samples_seconds':samples}


def interval_accounting(intervals, window_start, window_end):
    """Union and uncovered duration within one window; no causal attribution."""
    if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (window_start,window_end)) or window_end <= window_start:
        raise ValueError('window must be finite and have positive duration')
    clipped=[]
    for start,end in intervals:
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in (start,end)) or end < start:
            raise ValueError('intervals must be finite and ordered')
        lo=max(start,window_start);hi=min(end,window_end)
        if hi>lo:clipped.append((lo,hi))
    merged=[]
    for start,end in sorted(clipped):
        if merged and start <= merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    busy=sum(end-start for start,end in merged)
    return {'busy_duration':busy,'uncovered_duration':window_end-window_start-busy,
            'summed_clipped_duration':sum(end-start for start,end in clipped),
            'merged_intervals':merged,
            'interpretation':'same-scope union only; uncovered time does not establish cause'}


def benchmark(fn, repeats=7, warmup=2, clock=time.perf_counter):
    for name,value in [('repeats',repeats),('warmup',warmup)]:
        if isinstance(value,bool) or not isinstance(value,int) or value < (1 if name=='repeats' else 0):
            raise ValueError(name+' has an invalid count')
    for _ in range(warmup):fn()
    samples=[];checksum=None
    for _ in range(repeats):
        start=clock();checksum=fn();end=clock()
        samples.append(end-start)
    result=summarize(samples)
    result.update({'warmup_iterations':warmup,'result':checksum})
    return result


def workload(iterations):
    return sum((i*i)%104729 for i in range(iterations))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=20000)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2)
    args=parser.parse_args()
    if args.iterations<1:parser.error('iterations must be positive')
    try:result=benchmark(lambda:workload(args.iterations),args.repeats,args.warmup)
    except ValueError as error:parser.error(str(error))
    print(json.dumps({'schema_version':1,'lab':'L1','provenance':'measured CPU wall time',
        'environment':{'python':platform.python_version(),'platform':platform.platform()},
        'workload':{'kind':'Python scalar loop','iterations':args.iterations},
        'measurement':result,
        'constructed_interval_exercise':{'provenance':'analytical invented intervals; not a profile',
            'units':'milliseconds',**interval_accounting([(0,4),(2,7),(8,10)],0,10)}},indent=2,allow_nan=False))
