"""CPU execution-evidence protocol controls; no GPU or real sanitizer is invoked.

Only named, course-owned controls run here. Python in-process bookkeeping is not
a security boundary for hostile code. The nonce associates a record with a run;
it is not a signature or an authenticated hardware attestation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import secrets

from virtual_accelerator import ABI, ContractError, Descriptor, Device

TARGET = 'virtual-copy4-v1'
CONTROLS = ('legal', 'no_launch', 'missing_coverage', 'malformed_coverage',
            'replayed_coverage', 'wrong_result', 'dma_error', 'rejected_bounds',
            'missing_diagnostics')


def parse_record(raw):
    if not isinstance(raw, str):
        return None
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    try:
        record = json.loads(raw, object_pairs_hook=unique_pairs)
    except (ValueError, TypeError):
        return None
    return record if isinstance(record, dict) else None


def classify(coverage_raw, diagnostics_raw, nonce, correct):
    """Accept only harness-channel evidence for this exact one-command workload.

    Candidate output/stdout is deliberately not an argument. A production runner
    must establish who owns these channels; JSON schema alone does not do that.
    """
    coverage = parse_record(coverage_raw)
    valid = (isinstance(nonce, str) and bool(nonce)
             and coverage is not None and set(coverage) == {'nonce', 'target', 'started', 'completed'}
             and coverage['nonce'] == nonce and coverage['target'] == TARGET
             and type(coverage['started']) is int and coverage['started'] == 1
             and type(coverage['completed']) is int and coverage['completed'] == 1)
    diagnostics = parse_record(diagnostics_raw)
    checked = (diagnostics is not None
               and set(diagnostics) == {'nonce', 'check', 'findings'}
               and diagnostics['nonce'] == nonce
               and diagnostics['check'] == 'cpu-model-transfer-v1'
               and isinstance(diagnostics['findings'], list)
               and all(isinstance(x, str) and x for x in diagnostics['findings']))
    coverage_status = 'valid' if valid else 'invalid'
    diagnostics_status = ('passed' if not diagnostics['findings'] else 'failed') if checked else 'invalid'
    correctness_status = 'passed' if correct is True else 'failed'
    if not valid or not checked:
        status = 'invalid'
    elif diagnostics_status == 'failed' or correctness_status == 'failed':
        status = 'failed'
    else:
        status = 'passed'
    return dict(status=status, coverage=coverage_status, diagnostics=diagnostics_status,
                correctness=correctness_status, accepted=status == 'passed',
                scope='CPU functional protocol only')


def run_control(name):
    if name not in CONTROLS:
        raise ValueError('unknown course control')
    # The harness owns the model, nonce, trace, record construction and oracle.
    # No candidate-supplied launch count or stdout is parsed as evidence.
    nonce = secrets.token_hex(16)
    device = Device()
    device.boot()
    src, dst = device.alloc(4), device.alloc(4)
    device.write(src, bytes([0, 1, 254, 255]))
    findings = []
    claimed_stdout = ''
    output = None
    if name == 'no_launch':
        # Numerically right hardcoded output, with a fake candidate coverage claim.
        output = bytes([0, 1, 254, 255])
        claimed_stdout = '{"target_launches":1000,"status":"clean"}'
    else:
        try:
            descriptor = Descriptor(ABI, 1, 0, device.generation, src, dst,
                                    5 if name == 'rejected_bounds' else 4, 1)
            key = device.submit(descriptor.encode(), fault='dma_error' if name == 'dma_error' else 'none')
            status = device.wait(key, polling=True)
            device.reap(key)
            output = device.read(dst)
            if status != 'done':
                findings.append('modeled transfer completed with error')
        except ContractError as exc:
            findings.append(str(exc))
        if name == 'wrong_result':
            output = bytes([9, 9, 9, 9])
    device.assert_invariants()
    coverage = dict(nonce=nonce, target=TARGET,
                    started=sum(e['event'] == 'start' and e.get('cookie') == 1 for e in device.trace),
                    completed=sum(e['event'] == 'completion' and e.get('cookie') == 1 for e in device.trace))
    coverage_raw = json.dumps(coverage)
    diagnostics_raw = json.dumps(dict(nonce=nonce, check='cpu-model-transfer-v1', findings=findings))
    # Explicit harness-transport fault injection, after observing the real model run.
    if name == 'missing_coverage':
        coverage_raw = None
    elif name == 'malformed_coverage':
        coverage_raw = '{broken'
    elif name == 'replayed_coverage':
        coverage_raw = json.dumps(dict(coverage, nonce='previous-run'))
    elif name == 'missing_diagnostics':
        diagnostics_raw = None
    # Literal reference independent of the Device's copy implementation.
    correct = output == bytes([0, 1, 254, 255])
    return dict(control=name, nonce=nonce, target=TARGET,
                coverage_channel=coverage_raw, diagnostics_channel=diagnostics_raw,
                candidate_stdout_ignored=claimed_stdout,
                output=list(output) if output is not None else None,
                trace=device.trace, decision=classify(coverage_raw, diagnostics_raw, nonce, correct))


def report():
    rows = [run_control(name) for name in CONTROLS]
    return dict(schema=1, execution='cpu-functional-model', time_unit='model_tick',
                python=platform.python_version(),
                platform=platform.platform(), gpu='not_run', compute_sanitizer='not_run',
                security_scope='course-owned in-process controls; no hostile-code isolation',
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), Path(__file__).with_name('virtual_accelerator.py')]},
                counts={status: sum(r['decision']['status'] == status for r in rows)
                        for status in ('passed', 'failed', 'invalid')}, controls=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(report(), indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
