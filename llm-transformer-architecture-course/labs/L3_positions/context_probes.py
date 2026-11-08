"""Small controlled context fixtures, inspired by RULER task families.

Standard library only. Record counts are NOT model token counts. The program
validates fixture construction/oracles; it does not call or evaluate an LLM.
"""
import argparse
import json
import random
from pathlib import Path


POSITIONS = {'early': 0.1, 'middle': 0.5, 'late': 0.9}
TASKS = ('lookup', 'chain', 'count')


def evidence(task, seed, replica):
    rng = random.Random(seed * 1009 + replica)
    a, b, c = [f'k{n}' for n in rng.sample(range(100000, 999999), 3)]
    value = f'v{rng.randrange(10000000, 99999999)}'
    if task == 'lookup':
        return [f'VALUE {a} {value}'], f'Return the VALUE of {a}.', a, value
    if task == 'chain':
        return [f'LINK {a} {b}', f'LINK {b} {c}', f'VALUE {c} {value}'], \
               f'Follow exactly two LINK edges from {a}; return the VALUE at the destination.', a, value
    if task == 'count':
        count = 2 + replica % 3
        return [f'ITEM {a} item{i}' for i in range(count)], \
               f'How many ITEM records have key {a}? Return an integer.', a, str(count)
    raise ValueError('unsupported task')


def build_case(length, position, task, replica=0, seed=7):
    if isinstance(length, bool) or not isinstance(length, int) or length < 8:
        raise ValueError('length must be an integer of at least eight records')
    if position not in POSITIONS or replica < 0:
        raise ValueError('known position and nonnegative replica required')
    bundle, question, target, answer = evidence(task, seed, replica)
    start = int(POSITIONS[position] * (length - len(bundle)))
    rng = random.Random(seed * 991 + replica)
    filler = [f'DISTRACTOR noise{rng.randrange(1000000, 9999999)}' for _ in range(length-len(bundle))]
    records = filler[:start] + bundle + filler[start:]
    return {'id': f'{task}-n{length}-{position}-r{replica}', 'task': task,
            'length_records': length, 'position': position, 'replica': replica,
            'evidence_indices': list(range(start, start+len(bundle))),
            'records': records, 'question': question, 'target': target,
            'expected_answer': answer}


def prompt_for(case):
    """Only this context/question text goes to a future model adapter."""
    return 'Context:\n' + '\n'.join(case['records']) + '\nQuestion: ' + case['question']


def solve_fixture(case):
    """Debug oracle for this tiny grammar, not a model or comprehension score."""
    values, links, items = {}, {}, []
    for record in case['records']:
        parts = record.split()
        if parts[0] == 'DISTRACTOR':
            continue
        kind, key, value = parts
        if kind == 'VALUE': values[key] = value
        if kind == 'LINK': links[key] = value
        if kind == 'ITEM': items.append(key)
    key = case['target']
    if case['task'] == 'lookup': return values[key]
    if case['task'] == 'chain': return values[links[links[key]]]
    if case['task'] == 'count': return str(items.count(key))
    raise ValueError('unsupported task')


def score_answer(case, answer):
    return isinstance(answer, str) and answer.strip() == case['expected_answer']


def manifest(seed=7, replicas=3):
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError('positive integer replicas required')
    cases = [build_case(n, pos, task, r, seed) for n in (32, 64, 128)
             for pos in POSITIONS for task in TASKS for r in range(replicas)]
    return {'schema_version': 1, 'seed': seed, 'replicas_per_cell': replicas,
            'units': 'synthetic_records_not_model_tokens',
            'scope': 'fixture/oracle checks only; no LLM calls or quality measurements', 'cases': cases}


def summary(data):
    cases = data['cases']
    return {'case_count': len(cases), 'cells': len(cases)//data['replicas_per_cell'],
            'replicas_per_cell': data['replicas_per_cell'], 'units': data['units'],
            'oracle_agreement': sum(score_answer(c, solve_fixture(c)) for c in cases),
            'model_runs': 0, 'scope': data['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--replicas', type=int, default=3)
    parser.add_argument('--output', type=Path, help='New full-manifest JSON; refuses to overwrite')
    args = parser.parse_args()
    data = manifest(args.seed, args.replicas)
    result = summary(data)
    if args.output:
        with args.output.open('x') as stream: json.dump(data, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
