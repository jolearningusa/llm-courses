"""Bounded real two-rank CPU DDP/DCP restart; not TorchTitan/FSDP2 qualification."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_state_dict, set_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'distributed_resume_protocol.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for item in x: visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)  # Local visibility only: no crash/fsync/durability claim.


def rng_state():
    n = np.random.get_state()
    return {'torch': torch.get_rng_state().tolist(), 'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])]}


def restore_rng(value):
    def tuples(x): return tuple(tuples(v) for v in x) if isinstance(x, list) else x
    torch.set_rng_state(torch.tensor(value['torch'], dtype=torch.uint8))
    random.setstate(tuples(value['python']))
    n = value['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))


class App:
    def __init__(self, model, optimizer, omit_optimizer=False):
        self.model, self.optimizer, self.omit_optimizer = model, optimizer, omit_optimizer

    def state_dict(self):
        # DCP fills returned tensor storage in place. To omit optimizer restore,
        # exclude optimizer tensors from the load destination, not just its setter.
        if self.omit_optimizer:
            return {'model': get_model_state_dict(self.model)}
        model, optimizer = get_state_dict(self.model, self.optimizer)
        return {'model': model, 'optimizer': optimizer}

    def load_state_dict(self, state):
        if self.omit_optimizer:
            self.model.module.load_state_dict(state['model'])
        else:
            set_state_dict(self.model, self.optimizer, model_state_dict=state['model'],
                           optim_state_dict=state['optimizer'])


def verify_checkpoint(directory, config, runner_hash=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'complete.json').read_text())
    if manifest['world_size'] != config['world_size']: raise ValueError('world size changed')
    if manifest['protocol_sha256'] != sha(PROTOCOL): raise ValueError('protocol changed')
    if manifest['runner_sha256'] != (runner_hash or sha(__file__)): raise ValueError('runner changed')
    for rank in range(config['world_size']):
        if f'rank{rank}.json' not in manifest['files']: raise ValueError('rank state absent')
    if not any(p.startswith('dcp/') for p in manifest['files']): raise ValueError('DCP state absent')
    for name, expected in manifest['files'].items():
        if sha(directory / name) != expected: raise ValueError('checkpoint digest mismatch: ' + name)
    return manifest


def batch(cursor, rank, config):
    ids = [((cursor * config['world_size'] + rank) * config['batch_per_rank'] + j)
           for j in range(config['batch_per_rank'])]
    xs, ys = [], []
    for i in ids:
        generator = torch.Generator().manual_seed(10000 + i)
        x = torch.randn(4, dtype=torch.float64, generator=generator)
        y = torch.randn(2, dtype=torch.float64, generator=generator)
        xs.append(x + random.random() / 100 + np.random.uniform() / 100); ys.append(y)
    return torch.stack(xs), torch.stack(ys), ids


def worker(rank, config, phase, seed, output, checkpoint, rendezvous, mutation):
    started = time.perf_counter(); output, checkpoint = Path(output), Path(checkpoint)
    torch.set_num_threads(config['threads_per_rank'])
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=config['world_size'], timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(seed)
        model = DDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                    torch.nn.Dropout(.3), torch.nn.Linear(8, 2)).double())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        app = App(model, optimizer, mutation == 'omit_optimizer')
        torch.manual_seed(seed + rank + 100); random.seed(seed + rank + 200); np.random.seed(seed + rank + 300)
        cursor, completed, load_seconds, save_seconds = 0, 0, 0., 0.
        if phase == 'resume':
            tick = time.perf_counter(); verify_checkpoint(checkpoint, config)
            state = json.loads((checkpoint / f'rank{rank}.json').read_text())
            if state['seed'] != seed or state['rank'] != rank: raise ValueError('rank/seed changed')
            dcp.load({'app': app}, checkpoint_id=checkpoint / 'dcp')
            completed, cursor = state['completed_steps'], state['cursor']
            if mutation != 'omit_rng': restore_rng(state['rng'])
            if mutation == 'rewind_cursor': cursor -= 1
            dist.barrier(); load_seconds = time.perf_counter() - tick
        stop = config['checkpoint_after_steps'] if phase == 'prefix' else config['total_steps']
        trace = []
        while completed < stop:
            tick = time.perf_counter(); x, y, ids = batch(cursor, rank, config)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean(); loss.backward(); optimizer.step()
            completed += 1; cursor += 1
            trace.append({'completed_step': completed, 'ids': ids, 'input_sha256': tensor_hash([x,y]),
                          'loss': loss.item(), 'step_seconds': time.perf_counter() - tick})
        if phase == 'prefix':
            tick = time.perf_counter()
            if rank == 0: checkpoint.mkdir(parents=True, exist_ok=False)
            dist.barrier()
            dcp.save({'app': app}, checkpoint_id=checkpoint / 'dcp')
            write_json(checkpoint / f'rank{rank}.json', {'seed': seed, 'rank': rank,
                       'completed_steps': completed, 'cursor': cursor, 'rng': rng_state()})
            dist.barrier()
            if rank == 0:
                files = {str(p.relative_to(checkpoint)): sha(p) for p in sorted(checkpoint.rglob('*')) if p.is_file()}
                write_json(checkpoint / 'complete.json', {'world_size': config['world_size'],
                           'runner_sha256': sha(__file__), 'protocol_sha256': sha(PROTOCOL), 'files': files})
            dist.barrier(); save_seconds = time.perf_counter() - tick
        write_json(output / f'rank{rank}.json', {'phase': phase, 'seed': seed, 'rank': rank,
                   'mutation': mutation, 'completed_steps': completed, 'cursor': cursor, 'trace': trace,
                   'model_sha256': tensor_hash(model.module.state_dict()),
                   'optimizer_sha256': tensor_hash(optimizer.state_dict()),
                   'rng_sha256': hashlib.sha256(json.dumps(rng_state(),sort_keys=True).encode()).hexdigest(),
                   'save_stall_seconds': save_seconds, 'load_barrier_seconds': load_seconds,
                   'worker_seconds': time.perf_counter() - started})
    finally:
        dist.destroy_process_group()


def compare(baseline, prefix, resumed):
    original = baseline['trace']; joined = prefix['trace'] + resumed['trace']
    return {'loss_sequence_equal': [x['loss'] for x in original] == [x['loss'] for x in joined],
            'data_ids_equal': [x['ids'] for x in original] == [x['ids'] for x in joined],
            'input_hashes_equal': [x['input_sha256'] for x in original] == [x['input_sha256'] for x in joined],
            **{k + '_equal': baseline[k] == resumed[k] for k in
               ['completed_steps','cursor','model_sha256','optimizer_sha256','rng_sha256']}}


def artifact_controls(checkpoint, config, directory):
    results = {}
    for label in config['artifact_controls']:
        target = directory / label; shutil.copytree(checkpoint, target)
        conf = dict(config); runner = None
        if label == 'missing_completion': (target/'complete.json').unlink()
        elif label == 'missing_sidecar': (target/'rank1.json').unlink()
        elif label == 'tampered_dcp_file':
            p = next((target/'dcp').glob('*.distcp')); p.write_bytes(p.read_bytes() + b'corrupt')
        elif label == 'wrong_world_size': conf['world_size'] = 3
        elif label == 'wrong_protocol':
            m=json.loads((target/'complete.json').read_text());m['protocol_sha256']='bad';write_json(target/'complete.json',m)
        elif label == 'wrong_runner': runner='bad'
        try: verify_checkpoint(target, conf, runner)
        except (ValueError, FileNotFoundError) as error: results[label] = {'rejected':True,'reason':str(error).replace(str(directory),'<controls>')}
        else: raise AssertionError('invalid checkpoint accepted: ' + label)
    return results


def suite(directory):
    config = json.loads(PROTOCOL.read_text()); directory = Path(directory)
    if directory.exists(): raise FileExistsError('Preserve previous evidence; choose a new directory')
    directory.mkdir(parents=True)
    report = {'protocol_sha256':sha(PROTOCOL),'runner_sha256':sha(__file__),
              'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
              'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
              'world_size':config['world_size'],'seeds':[],'processes':[]}
    for name in ['torch','numpy']:
        if report[name] != config['runtime'][name]: raise RuntimeError('unqualified ' + name)
    for seed in config['seeds']:
        cp = directory / f'checkpoint-{seed}'
        cases = [('baseline','none'),('prefix','none'),('resume','none')]
        if seed == config['seeds'][0]: cases += [('resume',m) for m in config['negative_controls']]
        for phase, mutation in cases:
            target = directory / f'{seed}-{phase}-{mutation}'
            command = [sys.executable,__file__,'worker','--directory',str(target),'--checkpoint',str(cp),
                       '--phase',phase,'--seed',str(seed),'--mutation',mutation]
            tick=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,timeout=90)
            log=target.with_suffix('.log');log.write_text(result.stdout+result.stderr)
            report['processes'].append({'seed':seed,'phase':phase,'mutation':mutation,'exit_code':result.returncode,
                                        'fresh_process_seconds':time.perf_counter()-tick,'log':log.name})
            if result.returncode: raise RuntimeError('worker failed: '+str(log))
        ranks=[]
        for rank in range(config['world_size']):
            def read(phase, mutation='none'):
                return json.loads((directory/f'{seed}-{phase}-{mutation}'/f'rank{rank}.json').read_text())
            baseline,prefix,resumed=read('baseline'),read('prefix'),read('resume')
            checks=compare(baseline,prefix,resumed)
            if not all(checks.values()):raise AssertionError('full restore diverged: '+str(checks))
            negative={}
            if seed==config['seeds'][0]:
                for m in config['negative_controls']:
                    negative[m]=compare(baseline,prefix,read('resume',m))
                    if all(negative[m].values()):raise AssertionError('omission undetected: '+m)
            ranks.append({'rank':rank,'checks':checks,'negative_controls':negative,
                          'checkpoint_stall_seconds':prefix['save_stall_seconds'],
                          'load_barrier_seconds':resumed['load_barrier_seconds']})
        hashes=[json.loads((directory/f'{seed}-resume-none'/f'rank{r}.json').read_text())['model_sha256'] for r in range(config['world_size'])]
        if len(set(hashes))!=1:raise AssertionError('ranks disagree')
        report['seeds'].append({'seed':seed,'ranks':ranks,'final_model_ranks_equal':True,
                                'checkpoint_bytes':sum(p.stat().st_size for p in cp.rglob('*') if p.is_file()),
                                'unique_useful_examples':config['world_size']*config['total_steps']*config['batch_per_rank']})
    controls=directory/'artifact-controls';controls.mkdir()
    report['artifact_controls']=artifact_controls(directory/f"checkpoint-{config['seeds'][0]}",config,controls)
    shutil.rmtree(controls)
    write_json(directory/'summary.json',report)
    print('PASS: three seeds, two real ranks, full restore equality; three state omissions detected per rank; six artifact guards')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['suite','worker'])
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--phase',choices=['baseline','prefix','resume']);p.add_argument('--seed',type=int)
    p.add_argument('--mutation',default='none',choices=['none','omit_rng','omit_optimizer','rewind_cursor'])
    a=p.parse_args()
    if a.mode=='suite':suite(a.directory)
    else:
        a.directory.mkdir(parents=True,exist_ok=False)
        with tempfile.TemporaryDirectory(prefix='cmp5-rendezvous-') as temp:
            mp.spawn(worker,args=(json.loads(PROTOCOL.read_text()),a.phase,a.seed,str(a.directory),str(a.checkpoint),str(Path(temp)/'store'),a.mutation),nprocs=2,join=True)
