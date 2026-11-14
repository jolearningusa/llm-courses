"""Frozen local CPU producer/consumer pilot; preserves the historical L11 runner."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from deployment_lab import checked_forward
PROTOCOL = KIT / 'executorch_compat_protocol.json'
FIXTURE = KIT / 'executorch_compat_fixture.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(fixture, inputs):
    # Independent scalar double-precision evaluation of the frozen float32 values.
    result = []
    for row in inputs:
        hidden = [max(0.0, sum(a*b for a, b in zip(weights, row)) + bias)
                  for weights, bias in zip(fixture['weight1'], fixture['bias1'])]
        result.append([sum(a*b for a, b in zip(weights, hidden)) + bias
                       for weights, bias in zip(fixture['weight2'], fixture['bias2'])])
    return result


def fixture_model(fixture):
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(8,16), torch.nn.ReLU(), torch.nn.Linear(16,4)).eval()
    with torch.no_grad():
        for target, name in [(model[0].weight,'weight1'), (model[0].bias,'bias1'),
                             (model[2].weight,'weight2'), (model[2].bias,'bias2')]:
            target.copy_(torch.tensor(fixture[name], dtype=torch.float32))
    return model


def close(actual, expected, protocol):
    import torch
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32),
                               rtol=protocol['rtol'], atol=protocol['atol'])
    return (actual - torch.tensor(expected, dtype=torch.float32)).abs().max().item()


def input_guards():
    import torch
    # Independent spy proves malformed inputs never reach native execution.
    calls = []
    def spy(x): calls.append(True); return x
    cases = {'shape': torch.zeros(4,8), 'dtype': torch.zeros(3,8,dtype=torch.float64),
             'stride': torch.zeros(8,3).T}
    rows = {}
    for name, value in cases.items():
        try: checked_forward(spy, value)
        except ValueError as exc: rows[name] = str(exc)
        else: raise AssertionError('Invalid input accepted: ' + name)
    assert not calls
    return {'rejected': rows, 'native_calls_for_invalid_inputs': len(calls)}


def execute_checks(call, fixture, protocol):
    import torch
    rows = {}
    for name, values in fixture['inputs'].items():
        x = torch.tensor(values, dtype=torch.float32)
        start = time.perf_counter(); actual = checked_forward(call, x)
        first = time.perf_counter() - start
        error = close(actual, oracle(fixture, values), protocol)
        for _ in range(protocol['warmups']): checked_forward(call,x)
        warm = []
        for _ in range(protocol['warm_samples']):
            start = time.perf_counter(); actual = checked_forward(call,x)
            warm.append(time.perf_counter() - start)
        close(actual, oracle(fixture, values), protocol)
        rows[name] = {'max_abs_error_vs_scalar': error, 'output': actual.tolist(),
                      'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                      'first_case_call_seconds': first, 'warm_seconds': warm}
    return rows


def build(directory, fixture, protocol):
    import torch
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    if directory.exists(): raise FileExistsError('Use a new artifact directory')
    directory.mkdir(parents=True)
    bundled = Path(sys.executable).with_name('flatc')
    if not shutil.which('flatc') and bundled.exists(): os.environ['FLATC_EXECUTABLE'] = str(bundled)
    model = fixture_model(fixture)
    x = torch.tensor(fixture['inputs']['random'],dtype=torch.float32)
    with torch.inference_mode():
        eager = execute_checks(model, fixture, protocol)
    start = time.perf_counter(); exported = torch.export.export(model,(x,)); export_s = time.perf_counter()-start
    with torch.inference_mode():
        exported_checks = execute_checks(exported.module(), fixture, protocol)
    try: exported.module()(torch.zeros(4,8))
    except Exception as exc: export_rejection = {'type':type(exc).__name__,'message':str(exc)}
    else: raise AssertionError('Exported static shape accepted (4,8)')
    start = time.perf_counter()
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    graph = str(edge.exported_program().graph_module.graph)
    graph_calls = [str(n.target) for n in edge.exported_program().graph_module.graph.nodes if n.op=='call_function']
    program = edge.to_executorch()
    artifact = directory/'mlp.pte';artifact.write_bytes(program.buffer)
    lowering_s = time.perf_counter()-start
    delegates = [delegate.id for plan in program.executorch_program.execution_plan for delegate in plan.delegates]
    assert protocol['required_delegate'] in delegates, 'Required delegate absent'
    return {'artifact':'mlp.pte','artifact_sha256':sha(artifact),'artifact_bytes':artifact.stat().st_size,
            'eager':eager,'exported':exported_checks,'export_seconds':export_s,'lower_serialize_seconds':lowering_s,
            'export_shape_rejection':export_rejection,'delegates':delegates,'edge_graph':graph,
            'edge_call_targets':graph_calls,'fallback_aten_targets':[x for x in graph_calls if 'aten.' in x],
            'guards':input_guards()}


def load(directory, fixture, protocol):
    import torch
    from executorch.runtime import Runtime
    artifact=directory/'mlp.pte'
    start=time.perf_counter();program=Runtime.get().load_program(str(artifact));method=program.load_method('forward')
    elapsed=time.perf_counter()-start
    with torch.inference_mode(): checks=execute_checks(lambda value:method.execute([value])[0],fixture,protocol)
    return {'artifact_sha256':sha(artifact),'artifact_bytes':artifact.stat().st_size,
            'load_program_and_method_seconds':elapsed,'cases':checks,'guards':input_guards()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','load'])
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists(): raise FileExistsError('Preserve old report; choose a new output')
    start=time.perf_counter();stage='import'
    record={'phase':args.phase,'python':sys.version,'platform':platform.platform(),
            'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=KIT,text=True).strip(),
            'source_sha256':sha(Path(__file__)),'fixture_sha256':sha(FIXTURE),'protocol_sha256':sha(PROTOCOL),
            'historical_runner_sha256':sha(KIT/'deployment_lab.py'),
            'packages':sorted((d.metadata['Name'],d.version) for d in metadata.distributions())}
    try:
        import torch
        torch.set_num_threads(1)
        fixture=json.loads(FIXTURE.read_text());protocol=json.loads(PROTOCOL.read_text())
        versions={'executorch':metadata.version('executorch'),'torch':metadata.version('torch')}
        assert any(all(versions[k]==lane[k] for k in versions) for lane in [protocol['baseline'],protocol['candidate']])
        record.update(versions=versions,torch_threads=torch.get_num_threads(),device='cpu',dtype='float32',
                      venv_config=(Path(sys.prefix)/'pyvenv.cfg').read_text())
        stage=args.phase
        record['result']=build(args.directory,fixture,protocol) if args.phase=='build' else load(args.directory,fixture,protocol)
        record['passed']=True
    except Exception as exc:
        record.update(passed=False,stage=stage,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    record['phase_wall_seconds']=time.perf_counter()-start
    args.report.write_text(json.dumps(record,indent=2)+'\n')
    print(args.phase,record['passed'],args.report)
    sys.exit(0 if record['passed'] else 1)
