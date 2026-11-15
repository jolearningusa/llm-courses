"""One fresh-process worker or collector for the frozen L1 600-step table replay."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import resource
import statistics
import subprocess
import time

HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'table_replay_protocol.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_sources(protocol):
    if any(sha(HERE/name)!=h for name,h in protocol['source_sha256'].items()):
        raise ValueError('frozen training source changed')


def validate_run(row, protocol):
    key=(row['lane'],row['run'],row['seed'])
    if key[0] not in protocol['lanes'] or key[1] not in protocol['runs'] or key[2] not in protocol['seeds']:
        raise ValueError('unplanned run')
    expected=list(range(protocol['log_every'],protocol['steps']+1,protocol['log_every']))
    if [x[0] for x in row['curve']]!=expected:
        raise ValueError('missing curve steps')
    if not all(math.isfinite(x[1]) and math.isfinite(x[2]) for x in row['curve']):
        raise ValueError('nonfinite curve')
    if row['final']!=row['curve'][-1][2]:
        raise ValueError('final checkpoint mismatch')
    return key


def collect(rows, protocol):
    keys=[validate_run(r,protocol) for r in rows]
    planned={(lane,run,seed) for lane in protocol['lanes'] for run in protocol['runs'] for seed in protocol['seeds']}
    if len(keys)!=len(set(keys)) or set(keys)!=planned:
        raise ValueError('missing or duplicate runs')
    result={}
    for lane in protocol['lanes']:
        result[lane]={}
        for run in protocol['runs']:
            group=sorted([r for r in rows if r['lane']==lane and r['run']==run],key=lambda r:r['seed'])
            vals=[r['final'] for r in group];mean=statistics.mean(vals);sd=statistics.stdev(vals)
            result[lane][run]={'seeds':protocol['seeds'],'finals':vals,'mean':mean,'sample_sd':sd,
                              'rounded_3dp':[f'{mean:.3f}',f'{sd:.3f}'],
                              'historical_rounded_3dp':[f'{x:.3f}' for x in protocol['historical_600'][run]],
                              'mean_difference_from_rounded_historical':mean-protocol['historical_600'][run][0],
                              'rounded_mean_and_sd_match': [f'{mean:.3f}',f'{sd:.3f}']==[f'{x:.3f}' for x in protocol['historical_600'][run]],
                              'seconds':sum(r['train_seconds'] for r in group)}
    return result


def main():
    start=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['worker','collect']);p.add_argument('--lane');p.add_argument('--run');p.add_argument('--seed',type=int)
    p.add_argument('--directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(PROTOCOL.read_text());validate_sources(protocol)
    if a.phase=='collect':
        rows=[json.loads(x.read_text()) for x in sorted(a.directory.glob('torch*.json'))]
        for row in rows:
            if row['protocol_sha256']!=sha(PROTOCOL) or row['runner_sha256']!=sha(Path(__file__)):
                raise ValueError('worker source/protocol mismatch')
        report={'protocol':protocol,'protocol_sha256':sha(PROTOCOL),'runner_sha256':sha(Path(__file__)),
                'worker_files':{x.name:sha(x) for x in sorted(a.directory.glob('torch*.json'))},
                'results':collect(rows,protocol),'workers':len(rows),'training_steps':len(rows)*protocol['steps'],
                'summed_train_seconds':sum(r['train_seconds'] for r in rows)}
        with a.output.open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report['results'],indent=2));return
    if a.lane not in protocol['lanes'] or a.run not in protocol['runs'] or a.seed not in protocol['seeds']:
        raise ValueError('unplanned worker')
    with a.output.open('x') as f:
        import torch
        from train import train,make_text,VOCAB
        for name,want in protocol['lanes'][a.lane].items():
            if metadata.version(name)!=want:raise ValueError('wrong package cohort: '+name)
        torch.set_num_threads(protocol['threads']);torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        text_train=make_text(20000,1);text_val=make_text(2000,2)
        train_lines=re.findall(r'[^;]+;',text_train);val_lines=re.findall(r'[^;]+;',text_val)
        ready=time.perf_counter()
        final,curve=train(a.run,a.seed,steps=protocol['steps'],lr=protocol['lr'],wd=protocol['weight_decay'],
                          warmup_frac=protocol['warmup_frac'],log_every=protocol['log_every'])
        finish=time.perf_counter()
        row={'lane':a.lane,'run':a.run,'seed':a.seed,'final':final,'curve':curve,
             'protocol_sha256':sha(PROTOCOL),'runner_sha256':sha(Path(__file__)),
             'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
             'source_sha256':protocol['source_sha256'],'train_seconds':finish-ready,'in_worker_seconds':finish-start,
             'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'environment':{'python':platform.python_version(),'platform':platform.platform(),
                            'packages':dict(sorted((d.metadata['Name'],d.version) for d in metadata.distributions())),
                            'torch_config':torch.__config__.show(),'intraop_threads':torch.get_num_threads(),
                            'interop_threads':torch.get_num_interop_threads(),
                            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                            'default_dtype':str(torch.get_default_dtype()),'device':'cpu'},
             'data':{'train_sha256':hashlib.sha256(text_train.encode()).hexdigest(),
                     'validation_sha256':hashlib.sha256(text_val.encode()).hexdigest(),
                     'train_characters':len(text_train),'validation_characters':len(text_val),
                     'exact_unique_line_overlap':len(set(train_lines)&set(val_lines)),'vocab':VOCAB}}
        validate_run(row,protocol);json.dump(row,f,indent=2)
        print(a.lane,a.run,a.seed,final,finish-ready,flush=True)


if __name__=='__main__':main()
