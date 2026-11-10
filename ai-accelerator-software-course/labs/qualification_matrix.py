"""Scoped qualification teaching runner; CPU model only, no device adapter.

Records and hashes aid reproduction, not authentication. Never use this in-process
policy as an authorization boundary for untrusted candidates or device deployment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import py_compile
import tempfile
import uuid

import experiments


def decide(row, expected, required, scope, available=True):
    """Keep execution stage, check outcomes and evidence completeness separate.

    The caller owns expected identity and required checks. A candidate must not
    choose them. Evidence paths are references only; this does not verify logs.
    """
    if scope not in ('model', 'physical') or not required or len(set(required)) != len(required):
        raise ValueError('declare a scope and unique nonempty required checks')
    errors, failures, incomplete = [], [], []
    stage = row.get('stage')
    if stage not in ('built_only', 'simulator_run', 'device_run', 'not_run', 'unsupported'):
        errors.append('unknown execution stage')
    if (row.get('identity') != expected or not expected
            or set(expected) != {'source', 'build', 'target', 'run'}
            or (stage in ('built_only', 'simulator_run', 'device_run')
                and not all(expected.values()))):
        errors.append('source/build/target/run identity mismatch or missing')
    checks = row.get('checks', {})
    if not isinstance(checks, dict):
        checks = {}
        errors.append('malformed checks')
    if set(checks) != set(required):
        errors.append('required check set mismatch')
    for name, check in checks.items():
        if not isinstance(check, dict):
            errors.append(name + ': malformed outcome')
            continue
        status = check.get('status')
        if status == 'failed':
            failures.append(name)
        if status not in ('passed', 'failed', 'unsupported', 'not_run'):
            errors.append(name + ': unknown outcome')
        elif status in ('unsupported', 'not_run'):
            incomplete.append(name)
            if not check.get('reason'):
                errors.append(name + ': missing reason')
        elif not isinstance(check.get('evidence'), list) or not check['evidence'] or not all(
                isinstance(x, str) and x for x in check['evidence']):
            errors.append(name + ': missing evidence reference')
    if stage in ('built_only', 'not_run', 'unsupported'):
        incomplete.append('no workload execution')
    if stage in ('not_run', 'unsupported') and not row.get('reason'):
        errors.append('missing execution reason')
    if scope == 'physical' and (stage != 'device_run' or available is not True):
        incomplete.append('physical execution unavailable or absent')
    if scope == 'model' and stage != 'simulator_run':
        incomplete.append('model execution absent')
    status = ('invalid' if errors else 'failed' if failures else
              'incomplete' if incomplete else 'passed')
    return dict(status=status, accepted=status == 'passed', scope=scope,
                errors=errors, known_failures=failures, incomplete=incomplete)


def report():
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ('qualification_matrix.py', 'experiments.py', 'virtual_accelerator.py')}
    build_logs = []
    artifacts = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in sources:
            out = Path(directory) / (name + 'c')
            # Stable embedded filename; compile bytecode, not a target image.
            py_compile.compile(str(root / name), cfile=str(out), dfile=name,
                               doraise=True,
                               invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
            artifacts[name + 'c'] = hashlib.sha256(out.read_bytes()).hexdigest()
            build_logs.append(name + ': CPython bytecode compilation completed')
    source_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    build_id = hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode()).hexdigest()
    identity = dict(source=source_id, build=build_id, target='virtual-accelerator-model-v1',
                    run=uuid.uuid4().hex)
    raw = experiments.report()
    required = ['scenarios', 'invariants', 'bounded_control']
    built = dict(stage='built_only', identity=identity,
                 checks={name: dict(status='not_run', reason='build stage only') for name in required})
    model = dict(stage='simulator_run', identity=identity,
                 checks={name: dict(status=raw['checks'][name], evidence=['raw_model.checks.' + name])
                         for name in required})
    # No target image, board identity or measurement is inferred from CPU work.
    device_identity = dict(source=source_id, build=None, target=None, run=identity['run'])
    physical_required = ['device_functional', 'recovery', 'power_thermal']
    device = dict(stage='not_run', identity=device_identity,
                  reason='no designated board, supported adapter or target image',
                  checks={name: dict(status='not_run', reason='target path not executed')
                          for name in physical_required})
    return dict(schema=1, execution='CPU qualification-policy exercise',
                python=platform.python_version(), platform=platform.platform(),
                python_hash_seed=os.environ.get('PYTHONHASHSEED', 'unset'),
                source_sha256=sources, bytecode_sha256=artifacts, build_log=build_logs,
                required_model=required, required_physical=physical_required,
                rows=dict(build=built, model=model, physical=device), raw_model=raw,
                decisions=dict(build=decide(built, identity, required, 'model'),
                               model=decide(model, identity, required, 'model'),
                               physical=decide(device, device_identity, physical_required,
                                               'physical', available=False)),
                limits='No Twister, kernel, GPU, firmware, telemetry or HIL execution; '
                       'no log authentication, hostile-code isolation or deployment authorization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(report(), stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
