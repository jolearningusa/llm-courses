"""Analytical useful-work accounting with explicitly nonoverlapping phase costs."""
import json
import math


def amdahl(fraction,speedup):
    if not math.isfinite(fraction) or not 0<=fraction<=1 or not math.isfinite(speedup) or speedup<=0:raise ValueError('finite fraction [0,1] and positive speedup required')
    return 1/((1-fraction)+fraction/speedup)


def ledger(useful_tokens,phases,optimized_phase=None,speedup=1.):
    if useful_tokens<0 or not math.isfinite(useful_tokens) or not phases or any(not math.isfinite(t) or t<0 for t in phases.values()) or not math.isfinite(speedup) or speedup<=0:
        raise ValueError('finite nonnegative workload/time and positive speedup required')
    if optimized_phase is not None and optimized_phase not in phases:raise ValueError('unknown phase')
    elapsed=sum(t/(speedup if name==optimized_phase else 1) for name,t in phases.items())
    if elapsed<=0:raise ValueError('positive total time required')
    return {'provenance':'analytical constructed nonoverlapping phase ledger; no vendor benchmark','useful_tokens':useful_tokens,'elapsed_s':elapsed,'useful_tokens_per_s':useful_tokens/elapsed,'phases_s':{name:t/(speedup if name==optimized_phase else 1) for name,t in phases.items()},'scope':'phase intervals assumed disjoint; useful tokens exclude rejected/repeated work; no measured energy'}

if __name__=='__main__':
    phases={'useful_compute':60.,'communication':15.,'input':10.,'checkpoint':5.,'recovery':10.}
    print(json.dumps({'lab':'L15','baseline':ledger(12000,phases),'compute_2x':ledger(12000,phases,'useful_compute',2.),'amdahl_speedup':amdahl(.6,2.)},indent=2))
