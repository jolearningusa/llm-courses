"""Run unchanged lab tests in fresh workers; record one exact CPU package cohort.

This is a test-evidence runner, not a historical benchmark reproducer. See the
CPU lock evidence note for installation and platform boundaries.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import unittest

LABS = Path(__file__).resolve().parent
LOCK = LABS / 'requirements-cpu-macos-arm64-py312.lock'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def worker(lab, output):
    # One process per lab prevents collisions between modules named model/train.
    sys.path.insert(0, str(lab))
    import torch
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(lab), pattern='test*.py')
    ids = list(flatten(suite))
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'lab': lab.name, 'test_ids': ids, 'tests_run': result.testsRun,
              'failures': [(t.id(), message) for t, message in result.failures],
              'errors': [(t.id(), message) for t, message in result.errors],
              'skipped': [(t.id(), reason) for t, reason in result.skipped],
              'expected_failures': [t.id() for t, _ in result.expectedFailures],
              'unexpected_successes': [t.id() for t in result.unexpectedSuccesses],
              'elapsed_seconds': time.perf_counter() - start,
              'torch_threads': torch.get_num_threads(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    record['passed'] = (result.wasSuccessful() and result.testsRun == len(ids)
                        and len(ids) > 0 and len(set(ids)) == len(ids)
                        and not result.skipped and not result.expectedFailures)
    output.write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['passed'] else 1


def qualify(output):
    # Fail closed before running any suite when this specific cohort is absent.
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use an isolated venv, not the base interpreter')
    if (platform.python_version(), sys.platform, platform.machine()) != ('3.12.14', 'darwin', 'arm64'):
        raise RuntimeError('This lock was tested only on CPython 3.12.14 / macOS arm64')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
        raise RuntimeError('Unset PYTHONPATH and PYTHONHOME before qualification')
    expected = dict(re.findall(r'^([\w.-]+)==([^\s]+)', LOCK.read_text(), re.M))
    observed = {name: metadata.version(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f'Runtime package versions differ from lock: {observed}')
    if output.exists():
        raise FileExistsError('Choose a new evidence directory; prior results are preserved')
    output.mkdir(parents=True)
    import numpy as np
    import torch
    # Explicit bridge control: shared CPU storage in both directions, not imports alone.
    a = np.array([1., 2., 3.], dtype=np.float64)
    t = torch.from_numpy(a)
    t[1] = 7.
    assert a.tolist() == [1., 7., 3.]
    b = t.numpy()
    b[2] = 11.
    assert t.tolist() == [1., 7., 11.]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    check = subprocess.run([sys.executable, '-I', '-m', 'pip', 'check'], env=env,
                           capture_output=True, text=True)
    (output / 'pip-check.txt').write_text(check.stdout + check.stderr)
    record = {'schema': 1, 'source_revision': subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=LABS, text=True).strip(),
              'runner_sha256': digest(Path(__file__)), 'lock_sha256': digest(LOCK),
              'python': sys.version, 'implementation': platform.python_implementation(),
              'platform': platform.platform(), 'machine': platform.machine(),
              'venv': sys.prefix != sys.base_prefix,
              'include_system_site_packages': (Path(sys.prefix) / 'pyvenv.cfg').read_text(),
              'packages': observed, 'pip': metadata.version('pip'),
              'installed_distributions': sorted((d.metadata['Name'], d.version) for d in metadata.distributions()),
              'torch_build': torch.__config__.show(), 'numpy_build': np.show_config(mode='dicts'),
              'device': 'cpu', 'torch_numpy_shared_storage_control': 'passed',
              'worker_environment': {k: env[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONHASHSEED')},
              'rng_policy': 'Existing suite seeds unchanged; no additional Python/NumPy/Torch seed override',
              'pip_check_exit': check.returncode, 'labs': []}
    for lab in sorted(LABS.glob('L[0-9]_*')):
        if not lab.is_dir():
            continue
        source = {p.name: digest(p) for p in sorted(lab.glob('*.py'))}
        with (output / (lab.name + '.txt')).open('w') as log:
            proc = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()),
                                   '--worker', lab.name, '--output', str(output.resolve())],
                                  cwd=lab, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=600)
        result_file = output / (lab.name + '.json')
        row = json.loads(result_file.read_text()) if result_file.exists() else {'lab': lab.name, 'passed': False}
        row.update(exit_code=proc.returncode, source_sha256=source,
                   log=lab.name + '.txt', log_sha256=digest(output / (lab.name + '.txt')))
        record['labs'].append(row)
        print(lab.name, row.get('tests_run', 0), 'PASS' if row['passed'] else 'FAIL', flush=True)
    record['passed'] = (check.returncode == 0 and len(record['labs']) == 10
                        and all(r['passed'] and r['exit_code'] == 0 for r in record['labs']))
    record['total_tests'] = sum(r.get('tests_run', 0) for r in record['labs'])
    (output / 'manifest.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=[p.name for p in sorted(LABS.glob('L[0-9]_*')) if p.is_dir()])
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(LABS / args.worker, args.output / (args.worker + '.json')))
    sys.exit(qualify(args.output))
