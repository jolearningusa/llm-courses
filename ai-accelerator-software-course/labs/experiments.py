"""Executable CPU scenarios and bounded ownership-state exploration."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import random
from virtual_accelerator import ABI, ContractError, Descriptor, Device


def setup(fault='none', opcode=1):
    d = Device()
    d.boot()
    a, b = d.alloc(4), d.alloc(4)
    d.write(a, bytes([0, 1, 254, 255]))
    desc = Descriptor(ABI, opcode, 0, d.generation, a, b, 4, 1)
    return d, a, b, d.submit(desc.encode(), fault=fault)


def scenarios():
    results = []
    for name, opcode, fault in [('copy', 1, 'none'), ('add_one', 2, 'none'),
                                ('lost_irq', 1, 'drop_irq'), ('dma_error', 1, 'dma_error')]:
        d, a, b, key = setup(fault, opcode)
        if fault == 'drop_irq':
            try:
                d.wait(key, timeout=4)
                raise AssertionError('lost IRQ must time out without polling')
            except ContractError as exc:
                assert 'timeout' in str(exc)
            assert d.buffers[a].pins == 1
        status = d.wait(key, polling=True)
        d.reap(key)
        expected = bytes([1, 2, 255, 0]) if opcode == 2 else bytes([0, 1, 254, 255])
        if fault == 'dma_error':
            assert status == 'error' and d.read(b) == bytes(4)
        else:
            assert status == 'done' and d.read(b) == expected
        d.assert_invariants()
        results.append(dict(name=name, passed=True, registers=d.registers(), trace=d.trace))
    d, a, b, key = setup('hang')
    try:
        d.wait(key, timeout=7)
        raise AssertionError('hang must time out')
    except ContractError as exc:
        assert 'timeout' in str(exc)
    assert d.state == 'FAULT' and d.buffers[a].pins == 1
    d.reset()
    assert d.deliver_completion(key) is False
    d.boot()
    d.assert_invariants()
    results.append(dict(name='watchdog_reset_stale_completion', passed=True,
                        registers=d.registers(), trace=d.trace))
    return results


def explore(unsafe=False, depth=6):
    """Bounded abstract state search, not an ISA memory model/formal-tool proof.
    State: phase (new/queued/inflight/done), buffer_live. Releasing while queued
    or inflight violates the explicitly chosen outstanding-ownership invariant.
    """
    if type(depth) is not int or not 0 <= depth <= 20:
        raise ValueError('depth must be an integer in [0,20]')
    todo = deque([(('new', True), [])])
    seen = set()
    while todo:
        (phase, live), trace = todo.popleft()
        if (phase, live) in seen:
            continue
        seen.add((phase, live))
        if phase in ('queued', 'inflight') and not live:
            return dict(unsafe_policy=unsafe, depth_bound=depth, states=len(seen),
                        counterexample=trace, scope='abstract ownership invariant')
        if len(trace) >= depth:
            continue
        choices = []
        if phase == 'new' and live:
            choices.append(('submit', ('queued', live)))
        if phase == 'queued':
            choices += [('start', ('inflight', live)), ('cancel', ('done', live))]
        if phase == 'inflight':
            choices.append(('complete', ('done', live)))
        if live and (phase == 'done' or (unsafe and phase != 'new')):
            choices.append(('release', (phase, False)))
        for event, state in choices:
            todo.append((state, trace + [event]))
    return dict(unsafe_policy=unsafe, depth_bound=depth, states=len(seen),
                counterexample=None, scope='abstract ownership invariant')


def fuzz(seed=0, steps=200):
    """Seeded API/state exercise. Rejections are recorded, not test failures.
    This is not coverage-guided fuzzing and does not execute real driver calls.
    """
    if type(steps) is not int or not 0 <= steps <= 10000:
        raise ValueError('steps must be an integer in [0,10000]')
    rng = random.Random(seed)
    d = Device(capacity=3, memory_limit=64)
    accepted, rejected, actions, next_cookie = 0, 0, {}, 1
    d.boot()
    for _ in range(steps):
        op = rng.choice(['alloc', 'submit', 'tick', 'poll', 'reap', 'cancel', 'free', 'reset', 'boot'])
        actions[op] = actions.get(op, 0) + 1
        try:
            handles = list(d.buffers)
            keys = list(d.commands)
            if op == 'alloc':
                d.alloc(rng.choice([0, 4, 8, 128]))
            elif op == 'submit':
                if len(handles) < 2:
                    raise ContractError('fixture needs two buffers')
                a, b = rng.sample(handles, 2)
                desc = Descriptor(rng.choice([ABI, ABI, 255]), rng.choice([1, 2, 255]),
                                  0, d.generation, a, b, rng.choice([0, 4, 128]), next_cookie)
                next_cookie += 1
                d.submit(desc.encode(), fault=rng.choice(['none', 'drop_irq', 'hang', 'dma_error']))
            elif op == 'tick': d.tick()
            elif op == 'reset': d.reset()
            elif op == 'boot': d.boot()
            elif op == 'free':
                if not handles: raise ContractError('no buffer')
                d.free(rng.choice(handles))
            else:
                if not keys: raise ContractError('no command')
                getattr(d, op)(rng.choice(keys))
            accepted += 1
        except ContractError:
            rejected += 1
        d.assert_invariants()
    return dict(seed=seed, steps=steps, accepted=accepted, rejected=rejected,
                actions=actions, invariant_checks=steps)


def gate(record, scope='functional'):
    """Teaching decision only; metadata is not an authenticated attestation."""
    if scope != 'functional':
        return dict(accepted=False, reason='real target not qualified by CPU model')
    required = record.get('checks', {})
    keys = {'scenarios', 'invariants', 'bounded_control'}
    ok = (record.get('execution') == 'cpu-functional-model'
          and isinstance(required, dict) and set(required) == keys
          and all(v == 'passed' for v in required.values()))
    return dict(accepted=ok, reason='functional evidence only' if ok else 'missing/failing evidence')


def report():
    data = dict(schema=1, execution='cpu-functional-model', time_unit='model_tick',
                python=platform.python_version(), real_hardware='not_run',
                power_thermal='not_run', scenarios=scenarios(),
                exploration=[explore(False), explore(True)],
                fuzz=[fuzz(seed) for seed in range(10)])
    data['checks'] = {
        'scenarios': 'passed' if all(s['passed'] for s in data['scenarios']) else 'failed',
        'invariants': 'passed' if all(s['invariant_checks'] == s['steps'] for s in data['fuzz']) else 'failed',
        'bounded_control': 'passed' if (data['exploration'][0]['counterexample'] is None
            and data['exploration'][1]['counterexample'] == ['submit', 'release']) else 'failed'}
    data['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name('virtual_accelerator.py')]}
    data['functional_gate'] = gate(data)
    data['hardware_gate'] = gate(data, scope='hardware')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['scenarios', 'explore', 'fuzz', 'report'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'scenarios': data = scenarios()
    elif args.mode == 'explore': data = [explore(False), explore(True)]
    elif args.mode == 'fuzz': data = fuzz(args.seed, args.steps)
    else: data = report()
    text = json.dumps(data, indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream: stream.write(text)
    else: print(text, end='')


if __name__ == '__main__': main()
