"""Replay unchanged grpo.main in one worker; retain its printed precision only."""
import argparse
import contextlib
import hashlib
import importlib.metadata as metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'demo_replay_protocol.json'
PATTERN=re.compile(r'^(after SFT on 30 prompts|GRPO after 300 steps|Dr\. GRPO after 300 steps): mean reward ([0-9.]+) \(length: correct ([0-9.]+), incorrect ([0-9.]+)\)$')
METHODS=['SFT','GRPO','Dr. GRPO']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_stdout(text):
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if len(lines)!=3:raise ValueError('missing/extra demo result lines')
    rows={}
    for method,line in zip(METHODS,lines):
        m=PATTERN.fullmatch(line)
        if not m:raise ValueError('unexpected demo output')
        if not m[1].startswith('after SFT' if method=='SFT' else method+' after'):
            raise ValueError('wrong stage/order')
        values=list(map(float,m.groups()[1:]))
        if not all(math.isfinite(v) for v in values) or not 0<=values[0]<=1 or not all(0<=v<=8 for v in values[1:]):
            raise ValueError('invalid printed metric')
        rows[method]=values
    return rows


def collect(rows,protocol):
    keys=[(r['lane'],r['seed']) for r in rows]
    expected={(lane,seed) for lane in protocol['lanes'] for seed in protocol['seeds']}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('missing/duplicate/unplanned workers')
    for row in rows:
        if parse_stdout(row['stdout'])!=row['printed_metrics']:raise ValueError('metric/stdout mismatch')
    return {lane:{str(seed):next(r['printed_metrics'] for r in rows if r['lane']==lane and r['seed']==seed)
                  for seed in protocol['seeds']} for lane in protocol['lanes']}


def main():
    start=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['worker','collect']);p.add_argument('--lane');p.add_argument('--seed',type=int)
    p.add_argument('--directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(PROTOCOL.read_text())
    if any(sha(HERE/n)!=h for n,h in protocol['source_sha256'].items()):raise ValueError('frozen demo source changed')
    if a.phase=='collect':
        paths=sorted(a.directory.glob('torch*.json'));rows=[json.loads(x.read_text()) for x in paths]
        for row in rows:
            if row['protocol_sha256']!=sha(PROTOCOL) or row['runner_sha256']!=sha(Path(__file__)):
                raise ValueError('changed protocol/runner')
        metrics=collect(rows,protocol)
        result={'protocol':protocol,'results':metrics,'worker_files':{x.name:sha(x) for x in paths},
                'rounded_seed0_agreement':{lane:{m:metrics[lane]['0'][m]==protocol['historical_seed0'][m] for m in METHODS} for lane in protocol['lanes']},
                'summed_demo_seconds':sum(r['demo_seconds'] for r in rows),'workers':len(rows)}
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2));return
    if a.lane not in protocol['lanes'] or a.seed not in protocol['seeds']:raise ValueError('unplanned worker')
    with a.output.open('x') as f:
        import torch
        import grpo
        from task import prompts,split,MAX_LEN
        for n,v in protocol['lanes'][a.lane].items():
            if metadata.version(n)!=v:raise ValueError('wrong package cohort')
        torch.set_num_threads(protocol['intraop_threads']);torch.set_num_interop_threads(protocol['interop_threads'])
        sys.argv=['grpo.py','--steps',str(protocol['steps']),'--seed',str(a.seed)]
        stdout=io.StringIO();ready=time.perf_counter()
        with contextlib.redirect_stdout(stdout):grpo.main()
        finish=time.perf_counter();text=stdout.getvalue();metrics=parse_stdout(text)
        ps=prompts();sft_ps,_=split(a.seed)
        row={'lane':a.lane,'seed':a.seed,'stdout':text,'printed_metrics':metrics,
             'protocol_sha256':sha(PROTOCOL),'runner_sha256':sha(Path(__file__)),
             'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
             'source_sha256':protocol['source_sha256'],'demo_seconds':finish-ready,'worker_seconds':finish-start,
             'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':dict(sorted((d.metadata['Name'],d.version) for d in metadata.distributions())),
                            'torch_config':torch.__config__.show(),'intraop_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
                            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
             'sft_prompts':sft_ps,'rl_and_eval_prompts':ps,'max_len':MAX_LEN,'evaluation_seed':0,
             'evaluation_samples_per_prompt':8,'precision':'reward printed to 3dp, conditional lengths 2dp; no full-precision metrics or raw responses captured'}
        json.dump(row,f,indent=2);print(text,end='')


if __name__=='__main__':main()
