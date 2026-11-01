"""CPU teaching evaluator: numerical mutants and explicit queue-event traces.
No CUDA execution, arbitrary candidate loading, formal proof or device simulation.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path


def reference(x, eps=1e-5):
    if not x or not all(math.isfinite(v) for v in x) or eps <= 0:
        raise ValueError('nonempty finite input and positive epsilon required')
    scale = math.sqrt(math.fsum(v * v for v in x) / len(x) + eps)
    return [v / scale for v in x]


def reordered(x, eps=1e-5):
    scale = math.sqrt(sum(v * v for v in reversed(x)) / len(x) + eps)
    return [v / scale for v in x]


def zeros(x, eps=1e-5):
    return [0.0] * len(x)


def wrong_tail(x, eps=1e-5):
    y = reference(x, eps)
    return y[:-1] + [0.0]


def wrong_epsilon(x, eps=1e-5):
    scale = math.sqrt(sum(v * v for v in x) / len(x)) + eps
    return [v / scale for v in x]


CASES = {'ordinary': [1., -2., 3., -4.], 'tail': [1., 2., 4.],
         'near_zero': [1e-9, -1e-9, 2e-9], 'zeros': [0., 0., 0.],
         'outlier': [1e3, 1e-3, -1e-3, 2.], 'singleton': [-2.]}
CANDIDATES = {'reference': reference, 'reordered': reordered,
              'constant_zero': zeros, 'missing_tail': wrong_tail,
              'epsilon_outside_sqrt': wrong_epsilon}


def compare(expected, actual, atol=1e-8, rtol=1e-6):
    if atol < 0 or rtol < 0 or not math.isfinite(atol + rtol):
        raise ValueError('invalid tolerance')
    if len(expected) != len(actual):
        return False
    return all(math.isfinite(a) and math.isfinite(e)
               and abs(a-e) <= atol + rtol * abs(e)
               for e, a in zip(expected, actual))


def check_numerical(fn):
    results = []
    for name, x in CASES.items():
        try:
            y = fn(list(x))
            results.append({'case': name, 'pass': compare(reference(x), y)})
        except Exception as exc:
            results.append({'case': name, 'pass': False, 'error': type(exc).__name__})
    return {'accepted_on_fixtures': all(r['pass'] for r in results), 'cases': results}


def validate_trace(events):
    """Check one command per buffer in a declared trace, not all interleavings.

    submit pins a buffer; start makes it in-flight; completion/cancel/reset
    ends device ownership. release cannot race outstanding device work.
    Terminal event after reset must be rejected by generation.
    """
    generation = 0
    states = {}
    buffers = {}
    try:
        for i, event in enumerate(events):
            op = event['op']
            if op == 'reset':
                generation += 1
                for cmd in states:
                    states[cmd] = 'reset'
                buffers.clear()
                continue
            cmd = event['cmd']
            if event['generation'] != generation:
                raise ValueError('stale generation')
            if op == 'submit':
                buf = event['buffer']
                if cmd in states or buf in buffers:
                    raise ValueError('duplicate command or owned buffer')
                states[cmd] = 'queued'
                buffers[buf] = cmd
            elif op == 'start':
                if states.get(cmd) != 'queued':
                    raise ValueError('start before submit')
                states[cmd] = 'inflight'
            elif op == 'complete':
                if states.get(cmd) != 'inflight':
                    raise ValueError('completion without inflight work')
                states[cmd] = 'completed'
            elif op == 'cancel':
                if states.get(cmd) != 'queued':
                    raise ValueError('inflight work needs completion or reset')
                states[cmd] = 'cancelled'
            elif op == 'release':
                buf = event['buffer']
                if buffers.get(buf) != cmd or states.get(cmd) not in ('completed', 'cancelled'):
                    raise ValueError('release while device may own buffer')
                del buffers[buf]
            else:
                raise ValueError('unknown event')
        if buffers:
            raise ValueError('unreleased buffer at end of drained trace')
        return {'accepted_on_trace': True, 'generation': generation}
    except (ValueError, KeyError, TypeError) as exc:
        return {'accepted_on_trace': False, 'event_index': i, 'reason': str(exc)}


def fixture_traces():
    submit = {'op': 'submit', 'cmd': 'c0', 'generation': 0, 'buffer': 'b0'}
    start = {'op': 'start', 'cmd': 'c0', 'generation': 0}
    done = {'op': 'complete', 'cmd': 'c0', 'generation': 0}
    free = {'op': 'release', 'cmd': 'c0', 'generation': 0, 'buffer': 'b0'}
    return {'complete_then_release': [submit, start, done, free],
            'queued_cancel': [submit, {'op': 'cancel', 'cmd': 'c0', 'generation': 0}, free],
            'early_release': [submit, start, free, done],
            'old_completion_after_reset': [submit, start, {'op': 'reset'}, done],
            'reset_drains_ownership': [submit, start, {'op': 'reset'}]}


def report():
    return {'provenance': 'executed CPU numerical tests and authored event traces',
            'scope': 'fixtures only; no GPU, arbitrary-code isolation, or proof',
            'numerical': {name: dict(check_numerical(fn),
                source_sha256=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest())
                for name, fn in CANDIDATES.items()},
            'traces': {name: validate_trace(t) for name, t in fixture_traces().items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.dumps(report(), indent=2, sort_keys=True) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
