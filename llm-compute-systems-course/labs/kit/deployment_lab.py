"""Small CPU export/AOTInductor or ExecuTorch deployment experiment.
Separate environments: torch 2.9.1 baseline; ExecuTorch 1.0.0 + torch 2.9.1.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import platform
import time
import zipfile


def fixture():
    import torch
    torch.manual_seed(5)
    model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.ReLU(),torch.nn.Linear(16,4)).eval()
    return model,torch.randn(3,8)


def export_check(model,x):
    import torch
    tick=time.perf_counter()
    exported=torch.export.export(model,(x,))
    elapsed=time.perf_counter()-tick
    torch.testing.assert_close(exported.module()(x),model(x))
    try:exported.module()(torch.randn(4,8))
    except (RuntimeError,AssertionError):rejected=True
    else:rejected=False
    if not rejected:raise RuntimeError('static batch contract not enforced')
    return exported,elapsed


def checked_forward(call,x):
    import torch
    if x.device.type!='cpu' or x.dtype!=torch.float32 or tuple(x.shape)!=(3,8) or not x.is_contiguous():
        raise ValueError('this artifact contract requires contiguous CPU float32 (3,8)')
    return call(x)


def run(output,backend='aoti'):
    import torch
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,x=fixture()
    exported,export_s=export_check(model,x)
    with torch.no_grad():
        reference=model(x)
        torch.save({'input':x,'expected':reference},output/'fixture.pt')
        tick=time.perf_counter()
        if backend=='aoti':
            artifact=output/'mlp.pt2'
            torch._inductor.aoti_compile_and_package(exported,package_path=str(artifact))
            build_s=time.perf_counter()-tick
            tick=time.perf_counter()
            loaded=torch._inductor.aoti_load_package(str(artifact))
            load_s=time.perf_counter()-tick
            forward=lambda:checked_forward(loaded,x)
            with zipfile.ZipFile(artifact) as archive:
                extra={'archive_entries':archive.namelist()}
        elif backend=='executorch':
            from importlib.metadata import version
            bundled_flatc=Path(sys.executable).with_name('flatc')
            if not shutil.which('flatc') and bundled_flatc.exists():
                os.environ.setdefault('FLATC_EXECUTABLE',str(bundled_flatc))
            from executorch.exir import to_edge_transform_and_lower
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
            from executorch.runtime import Runtime
            program=to_edge_transform_and_lower(exported,partitioner=[XnnpackPartitioner()]).to_executorch()
            artifact=output/'mlp.pte'
            artifact.write_bytes(program.buffer)
            build_s=time.perf_counter()-tick
            tick=time.perf_counter()
            loaded=Runtime.get().load_program(str(artifact)).load_method('forward')
            load_s=time.perf_counter()-tick
            forward=lambda:checked_forward(lambda value:loaded.execute([value])[0],x)
            extra={'executorch':version('executorch'),'delegate_requested':'XNNPACK','delegate_segments':len(program.executorch_program.backend_delegate_data)}
        else:raise ValueError('unknown backend')
        tick=time.perf_counter();actual=forward();first_s=time.perf_counter()-tick
        torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
        for _ in range(3):forward()
        samples=[]
        for _ in range(10):
            tick=time.perf_counter();actual=forward();samples.append(time.perf_counter()-tick)
        torch.testing.assert_close(actual,reference,rtol=1e-5,atol=1e-6)
        baselines={}
        for label,call in [('eager',lambda:model(x)),('exported',lambda:exported.module()(x))]:
            for _ in range(3):call()
            raw=[]
            for _ in range(10):
                tick=time.perf_counter();value=call();raw.append(time.perf_counter()-tick)
            torch.testing.assert_close(value,reference)
            baselines[label]=raw
    return {'lab':'L14','provenance':'measured local CPU artifact build/load/execute; small synthetic MLP, not LLM performance','backend':backend,'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'input_shape':list(x.shape),'contract':'contiguous static CPU float32 shape (3,8); export rejects (4,8); checked loader rejects other shape/dtype/stride','export_seconds':export_s,'build_seconds':build_s,'load_seconds':load_s,'first_execution_seconds':first_s,'warm_wall_seconds':samples,'baseline_wall_seconds':baselines,'artifact':artifact.name,'artifact_bytes':artifact.stat().st_size,'max_abs_error':(actual-reference).abs().max().item(),**extra}

def load_existing(output,backend):
    import torch
    from importlib.metadata import version
    output=Path(output)
    fixture_data=torch.load(output/'fixture.pt',weights_only=True)
    x,expected=fixture_data['input'],fixture_data['expected']
    tick=time.perf_counter()
    if backend=='aoti':
        call=torch._inductor.aoti_load_package(str(output/'mlp.pt2'))
    else:
        from executorch.runtime import Runtime
        method=Runtime.get().load_program(str(output/'mlp.pte')).load_method('forward')
        call=lambda value:method.execute([value])[0]
    load_s=time.perf_counter()-tick
    actual=checked_forward(call,x)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    try:checked_forward(call,torch.zeros(4,8))
    except ValueError:pass
    else:raise RuntimeError('loader guard failed')
    return {'lab':'L14','mode':'load existing artifact without model construction','backend':backend,'torch':torch.__version__,'load_seconds':load_s,'max_abs_error':(actual-expected).abs().max().item(),'bad_shape_rejected':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['aoti','executorch'],default='aoti')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--load-existing',action='store_true')
    args=parser.parse_args()
    print(json.dumps(load_existing(args.output,args.backend) if args.load_existing else run(args.output,args.backend),indent=2))
