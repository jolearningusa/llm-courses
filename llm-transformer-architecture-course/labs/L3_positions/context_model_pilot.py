"""Frozen optional local-model adapter for the existing ARC-3 record fixtures.

freeze tokenizes/plans without inference; run verifies that artifact before calls.
Labels, case IDs and grading metadata do not enter the model message builder.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time

from context_probes import build_case, manifest, prompt_for, score_answer, solve_fixture

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'context_model_protocol.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def cases(protocol):
    grid = manifest(seed=protocol['fixture_seed'], replicas=protocol['replicas'])['cases']
    if sorted({c['length_records'] for c in grid}) != protocol['record_lengths']:
        raise ValueError('fixture lengths changed')
    if sorted({c['task'] for c in grid}) != sorted(protocol['tasks']):
        raise ValueError('fixture tasks changed')
    if sorted({c['position'] for c in grid}) != sorted(protocol['positions']):
        raise ValueError('fixture positions changed')
    short = []
    for task in protocol['tasks']:
        for replica in range(protocol['replicas']):
            case = build_case(32, 'early', task, replica, protocol['fixture_seed'])
            case['records'] = [case['records'][i] for i in case['evidence_indices']]
            case.update(id=f'evidence-only-{task}-r{replica}', position='evidence-only',
                        length_records=len(case['records']), evidence_indices=list(range(len(case['records']))))
            short.append(case)
    # Fixed shuffle reduces fixed cell/order alignment; it is not timing replication.
    random.Random(protocol['seed']).shuffle(grid)
    return short + grid


def messages(records, question, system):
    # No expected_answer, target or case id accepted at this boundary.
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': 'Context:\n' + '\n'.join(records) + '\nQuestion: ' + question}]


def evidence_spans(prompt, records, indices, offsets):
    marker = 'Context:\n'
    base = prompt.index(marker) + len(marker)
    ranges = []
    for i in indices:
        start = base + sum(len(r) + 1 for r in records[:i])
        end = start + len(records[i])
        if prompt[start:end] != records[i]:
            raise ValueError('evidence not found at constructed boundary')
        tokens = [j for j, (a, b) in enumerate(offsets) if b > start and a < end]
        if not tokens:
            raise ValueError('evidence token span missing')
        ranges.append({'record_index': i, 'char_start': start, 'char_end': end,
                       'token_start': tokens[0], 'token_end_exclusive': tokens[-1] + 1})
    return ranges


def eligible(count, protocol, model_window):
    return count <= protocol['input_cap'] and count + protocol['max_new_tokens'] <= model_window


def identity(model_dir, protocol):
    observed = {name: sha(model_dir / name) for name in protocol['model_files']}
    if observed != protocol['model_files']:
        raise ValueError('pinned model/tokenizer identity changed')
    return observed


def sources():
    return {p.name: sha(p) for p in [Path(__file__), PROTOCOL, HERE / 'context_probes.py']}


def prepare(model_dir, protocol):
    from transformers import AutoTokenizer
    model_files = identity(model_dir, protocol)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    window = json.loads((model_dir / 'config.json').read_text())['max_position_embeddings']
    rows = []
    for case in cases(protocol):
        assert score_answer(case, solve_fixture(case))
        prompt = tokenizer.apply_chat_template(messages(case['records'], case['question'], protocol['system']),
                                                tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokens.pop('offset_mapping')
        n = len(tokens['input_ids'])
        rows.append({'case': case, 'prompt': prompt, 'input_ids_sha256': digest(tokens['input_ids']),
                     'input_tokens': n, 'eligible': eligible(n, protocol, window),
                     'evidence_spans': evidence_spans(prompt, case['records'], case['evidence_indices'], offsets)})
    return tokenizer, {'sources': sources(), 'model_files': model_files, 'configured_window': window,
                       'case_count': len(rows), 'rows': rows}


def validate_frozen(observed, frozen):
    if observed != frozen:
        raise ValueError('frozen plan/source/model changed; no generation allowed')


def summarize(rows):
    cells = {}
    for row in rows:
        case = row['case']
        key = f"{case['length_records']}:{case['position']}:{case['task']}"
        cell = cells.setdefault(key, {'planned': 0, 'eligible': 0, 'completed': 0,
                                     'correct': 0, 'errors': 0, 'cap_hits': 0,
                                     'input_tokens': [], 'output_tokens': 0, 'seconds': 0.0})
        cell['planned'] += 1
        cell['eligible'] += row['eligible']
        cell['input_tokens'].append(row['input_tokens'])
        cell['completed'] += row['status'] == 'completed'
        cell['errors'] += row['status'] == 'error'
        cell['correct'] += row.get('correct', False)
        cell['cap_hits'] += row.get('hit_output_cap_without_eos', False)
        cell['output_tokens'] += len(row.get('output_ids', []))
        cell['seconds'] += row.get('generate_seconds', 0)
    return cells


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'run'])
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    protocol = json.loads(PROTOCOL.read_text())
    for name in ('torch', 'numpy', 'transformers'):
        if metadata.version(name) != protocol[name]:
            raise ValueError('unqualified package version: ' + name)
    tokenizer, plan = prepare(args.model_dir, protocol)
    if args.phase == 'freeze':
        with args.plan.open('x') as stream:
            json.dump(plan, stream, indent=2)
        print(json.dumps({'planned': len(plan['rows']), 'eligible': sum(r['eligible'] for r in plan['rows']),
                          'sources': plan['sources'], 'model_generations': 0}, indent=2))
        return
    validate_frozen(plan, json.loads(args.plan.read_text()))
    if args.output is None:
        raise ValueError('output path required')
    # Claim the new path before inference; retain each completed/error row incrementally.
    with args.output.open('x') as stream:
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM
        torch.set_num_threads(protocol['torch_threads'])
        torch.manual_seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        ready = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.float32,
                    attn_implementation=protocol['attention']).to('cpu').eval()
        loaded = time.perf_counter()
        if model.config.rope_scaling is not None:
            raise ValueError('unexpected positional scaling')
        result = {'protocol': protocol, 'plan_sha256': sha(args.plan), 'sources': sources(),
                  'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
                  'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                  'packages': dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions())),
                                  'torch_threads': torch.get_num_threads(), 'device': 'cpu', 'dtype': 'float32',
                                  'attention': model.config._attn_implementation,
                                  'rope_theta': model.config.rope_theta, 'rope_scaling': model.config.rope_scaling},
                  'prepare_seconds': ready - start, 'load_seconds': loaded - ready, 'rows': []}
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        # No warmup: all 90 planned generations are retained, with order exposed.
        for index, planned in enumerate(plan['rows']):
            row = dict(planned, execution_index=index)
            if not row['eligible']:
                row.update(status='ineligible', reason='input_cap_or_output_reserve')
            else:
                try:
                    inputs = tokenizer(row['prompt'], add_special_tokens=False, return_tensors='pt')
                    assert digest(inputs['input_ids'][0].tolist()) == row['input_ids_sha256']
                    call_start = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=protocol['max_new_tokens'],
                            do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                    elapsed = time.perf_counter() - call_start
                    ids = output[0, row['input_tokens']:].tolist()
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    row.update(status='completed', output_ids=ids, text=text,
                               correct=score_answer(row['case'], text), generate_seconds=elapsed,
                               hit_output_cap_without_eos=len(ids) == protocol['max_new_tokens'] and ids[-1] not in eos)
                except Exception as error:
                    row.update(status='error', error_type=type(error).__name__, error=str(error))
            result['rows'].append(row)
            result['cells'] = summarize(result['rows'])
            result['elapsed_seconds'] = time.perf_counter() - start
            result['peak_process_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result['rss_units'] = 'macOS bytes; process lifetime includes imports/load; not per-cell or device memory'
            stream.seek(0); stream.truncate(); json.dump(result, stream, indent=2); stream.flush()
            print(index + 1, row['case']['id'], row['status'], row.get('correct'), flush=True)
    print(json.dumps({'completed': sum(r['status'] == 'completed' for r in result['rows']),
                      'correct': sum(r.get('correct', False) for r in result['rows']),
                      'elapsed_seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    main()
