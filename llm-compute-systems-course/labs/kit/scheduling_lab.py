"""Deterministic virtual-time scheduling exercise, NOT measured inference."""
import argparse
import json
import math
from pathlib import Path


def transfer_seconds(state_bytes, bandwidth_bytes_s, overhead_s=0):
    if state_bytes<0 or bandwidth_bytes_s<=0 or overhead_s<0:raise ValueError('invalid transfer assumptions')
    return state_bytes/bandwidth_bytes_s+overhead_s


def replay(workload, chunk=32, prefill_s_per_token=0.0002, decode_base_s=0.002, decode_s_per_request=0.0001, iteration_s=0.0003):
    if not workload or chunk<1 or any(x<0 for x in (prefill_s_per_token,decode_base_s,decode_s_per_request,iteration_s)) or prefill_s_per_token+decode_base_s+decode_s_per_request+iteration_s==0:
        raise ValueError('nonempty workload and positive progress costs required')
    if len({r['id'] for r in workload})!=len(workload):raise ValueError('duplicate request IDs')
    pending=[]
    for r in workload:
        if not math.isfinite(r['arrival_s']) or r['arrival_s']<0 or any(not isinstance(r[k],int) or r[k]<1 for k in ('prompt_tokens','output_tokens')):raise ValueError('invalid request')
        pending.append(dict(r,remaining_prompt=r['prompt_tokens'],emitted=[]))
    pending.sort(key=lambda r:r['arrival_s'])
    active=[];finished=[];now=0.;iterations=0
    while pending or active:
        if not active and pending:now=max(now,pending[0]['arrival_s'])
        while pending and pending[0]['arrival_s']<=now:active.append(pending.pop(0))
        decoding=[r for r in active if r['remaining_prompt']==0]
        budget=chunk;prefilled=0
        for r in active:
            take=min(r['remaining_prompt'],budget)
            r['remaining_prompt']-=take;budget-=take;prefilled+=take
            if budget==0:break
        cost=iteration_s+prefilled*prefill_s_per_token
        if decoding:cost+=decode_base_s+decode_s_per_request*len(decoding)
        now+=cost;iterations+=1
        for r in decoding:r['emitted'].append(now)
        for r in list(active):
            if len(r['emitted'])==r['output_tokens']:
                finished.append(r);active.remove(r)
    rows=[]
    for r in finished:
        times=r['emitted']
        rows.append({'id':r['id'],'arrival_s':r['arrival_s'],'ttft_s':times[0]-r['arrival_s'],'completion_s':times[-1]-r['arrival_s'],'tpot_s':(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None,'emitted_tokens':len(times)})
    return {'provenance':'virtual-time simulation with constructed costs, no model or device','resources':'one simulated colocated engine for both policies','chunk_tokens':chunk,'cost_assumptions':{'prefill_s_per_token':prefill_s_per_token,'decode_base_s':decode_base_s,'decode_s_per_request':decode_s_per_request,'iteration_s':iteration_s},'iterations':iterations,'elapsed_s':now-min(r['arrival_s'] for r in workload),'requests':rows,'completed':len(rows),'failed':0,'rejected':0,'timeouts':0,'scope':'finite workload drained; no KV capacity, cancellations or failure model'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workload',type=Path)
    args=parser.parse_args()
    workload=json.loads(args.workload.read_text()) if args.workload else [{'id':'short','arrival_s':0.,'prompt_tokens':8,'output_tokens':8},{'id':'long','arrival_s':0.005,'prompt_tokens':512,'output_tokens':4},{'id':'late','arrival_s':0.02,'prompt_tokens':16,'output_tokens':4}]
    print(json.dumps({'lab':'L12','runs':[replay(workload,32),replay(workload,10**9)],'transfer_example':{'state_bytes':512*1024**2,'bandwidth_bytes_s':50e9,'overhead_s':0.001,'seconds':transfer_seconds(512*1024**2,50e9,0.001),'provenance':'analytical ideal effective bandwidth assumption'}},indent=2))
if __name__=='__main__':main()
