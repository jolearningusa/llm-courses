"""One frozen local four-condition memory pilot; no kit TODO implementations.

Write reads histories only. Answer loads the committed write artifact and keeps
labels out of prompt/retrieval construction. All generation is local and greedy.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import re
import string
import subprocess
import sys
import time

KIT = Path(__file__).resolve().parent
PROTOCOL = KIT / 'memory_pilot_protocol.json'
HISTORIES = KIT / 'tasks/memory_pilot/histories.json'
QUESTIONS = KIT / 'tasks/memory_pilot/questions.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize(text):
    return ' '.join(text.casefold().split()).strip(string.punctuation + ' ')


def score(text, answer):
    return normalize(text) == normalize(answer)


def retrieve(chunks, query, protocol):
    def words(text):
        return set(re.findall(r'[a-z0-9]+', text.casefold())) - set(protocol['stopwords'])
    q = words(query)
    ranked = sorted(enumerate(chunks), key=lambda row: (-len(q & words(row[1])), row[0]))
    return [{'index': i, 'text': text, 'score': len(q & words(text))}
            for i, text in ranked[:protocol['top_k']]]


def context_for(condition, history, query, writes, protocol):
    if condition == 'none':
        return protocol['empty_memory'], []
    if condition == 'full':
        return '\n'.join(history['turns']), []
    if condition not in ('verbatim', 'extracted'):
        raise ValueError(condition)
    chunks = history['turns'] if condition == 'verbatim' else writes[history['id']]['chunks']
    selected = retrieve(chunks, query, protocol)
    return '\n'.join(row['text'] for row in selected) or protocol['empty_memory'], selected


def answer_messages(condition, history, query, writes, protocol):
    memory, selected = context_for(condition, history, query, writes, protocol)
    return [{'role': 'system', 'content': protocol['answer_system']},
            {'role': 'user', 'content': protocol['answer_user_template'].format(memory=memory, question=query)}], selected


def identity(model_dir):
    return {p.name: sha(p) for p in sorted(model_dir.iterdir()) if p.is_file()}


class LocalGenerator:
    def __init__(self, model_dir, protocol):
        import torch
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for name in ('torch', 'numpy', 'transformers'):
            if metadata.version(name) != protocol[name]:
                raise RuntimeError('Unqualified package version: ' + name)
        torch.set_num_threads(protocol['torch_threads'])
        torch.manual_seed(protocol['seed'])
        np.random.seed(protocol['seed'])
        self.torch, self.protocol = torch, protocol
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation=protocol['attention']).to('cpu').eval()
        self.load_seconds = time.perf_counter() - start
        self.warmup = self.generate([{'role': 'user', 'content': 'What is 1 plus 1?'}], 1)

    def generate(self, messages, limit):
        start = time.perf_counter()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        count = inputs['input_ids'].shape[1]
        if count > self.protocol['input_cap']:
            return {'eligible': False, 'reason': 'input_cap', 'input_tokens': count,
                    'prompt': prompt, 'input_cap': self.protocol['input_cap'],
                    'elapsed_seconds': time.perf_counter() - start}
        ready = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                         num_beams=1, use_cache=self.protocol['use_cache'],
                                         pad_token_id=self.tokenizer.eos_token_id)
        generated = output[0, count:].tolist()
        completed = time.perf_counter()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {'eligible': True, 'input_tokens': count, 'output_tokens': len(generated),
                'input_ids_sha256': object_sha(inputs['input_ids'][0].tolist()),
                'output_ids': generated, 'prompt': prompt, 'text': text,
                'hit_output_cap_without_eos': len(generated) == limit and (not generated or generated[-1] not in eos),
                'tokenize_seconds': ready - start, 'generate_seconds': completed - ready,
                'elapsed_seconds': time.perf_counter() - start}


def base_record(protocol, model_dir, generator):
    return {'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=KIT, text=True).strip(),
            'runner_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PROTOCOL),
            'histories_sha256': sha(HISTORIES), 'model_id': protocol['model_id'],
            'model_revision': protocol['model_revision'], 'model_files_sha256': identity(model_dir),
            'python': sys.version, 'platform': platform.platform(),
            'packages': sorted((d.metadata['Name'], d.version) for d in metadata.distributions()),
            'torch_threads': generator.torch.get_num_threads(), 'device': 'cpu', 'dtype': 'float32',
            'torch_deterministic_algorithms': generator.torch.are_deterministic_algorithms_enabled(),
            'model_load_seconds': generator.load_seconds, 'warmup': generator.warmup}


def write_phase(protocol, model_dir, generator):
    # Intentionally does not read QUESTIONS; extraction cannot see QA labels or queries.
    histories = json.loads(HISTORIES.read_text())
    record = base_record(protocol, model_dir, generator)
    writes = []
    for h in histories:
        start = time.perf_counter()
        generated = generator.generate([{'role': 'system', 'content': protocol['write_system']},
                                        {'role': 'user', 'content': '\n'.join(h['turns'])}],
                                       protocol['write_max_new_tokens'])
        chunks = [line.strip() for line in generated.get('text', '').splitlines() if line.strip()]
        writes.append({'history_id': h['id'], 'generation': generated, 'chunks': chunks,
                       'store_sha256': object_sha(chunks), 'write_seconds': time.perf_counter() - start})
        print('write', h['id'], 'retained', len(chunks), 'generated chunks', flush=True)
    record['writes'] = writes
    return record


def verify_writes(record, protocol, model_dir):
    if record['protocol_sha256'] != sha(PROTOCOL) or record['histories_sha256'] != sha(HISTORIES):
        raise ValueError('Write inputs differ from frozen protocol/history')
    if record['model_files_sha256'] != identity(model_dir) or record['model_revision'] != protocol['model_revision']:
        raise ValueError('Write model identity mismatch')
    if record['runner_sha256'] != sha(Path(__file__)):
        raise ValueError('Writer/answer runner mismatch')
    for row in record['writes']:
        if row['store_sha256'] != object_sha(row['chunks']):
            raise ValueError('Write store mutation')
    return {row['history_id']: row for row in record['writes']}


def answer_phase(protocol, model_dir, generator, writes_file):
    write_record = json.loads(writes_file.read_text())
    stores = verify_writes(write_record, protocol, model_dir)
    histories = {h['id']: h for h in json.loads(HISTORIES.read_text())}
    questions = json.loads(QUESTIONS.read_text())
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    record = base_record(protocol, model_dir, generator)
    record.update(questions_sha256=sha(QUESTIONS), writes_sha256=sha(writes_file), rows=[])
    for index, q in enumerate(questions):
        conditions = protocol['conditions']
        offset = index % len(conditions)
        for condition in conditions[offset:] + conditions[:offset]:
            start = time.perf_counter()
            messages, selected = answer_messages(condition, histories[q['history_id']], q['question'],
                                                 stores, protocol)
            prepared = time.perf_counter()
            # No answer label is supplied to prompt or retrieval construction.
            try:
                generated = generator.generate(messages, protocol['answer_max_new_tokens'])
                error = None
            except Exception as exc:
                generated = {'eligible': True, 'text': '', 'input_tokens': None, 'output_tokens': None}
                error = type(exc).__name__ + ': ' + str(exc)
            row = {'question_id': q['id'], 'history_id': q['history_id'], 'type': q['type'],
                   'condition': condition, 'selected': selected, 'generation': generated,
                   'reference': q['answer'], 'error': error,
                   'correct': generated['eligible'] and error is None and score(generated['text'], q['answer']),
                   'prepare_seconds': prepared - start, 'read_answer_seconds': time.perf_counter() - start}
            record['rows'].append(row)
            print(q['id'], condition, 'recorded', flush=True)
    record['summary'] = summarize(record['rows'], write_record, protocol)
    return record


def summarize(rows, write_record, protocol):
    summary = {}
    for condition in protocol['conditions']:
        chosen = [r for r in rows if r['condition'] == condition]
        eligible = [r for r in chosen if r['generation']['eligible']]
        types = sorted({r['type'] for r in rows})
        write = write_record['writes'] if condition == 'extracted' else []
        # Verbatim's actual construction is a list of seven existing turns, no model/embedding call.
        summary[condition] = {'planned': len(chosen), 'eligible': len(eligible),
                              'correct': sum(r['correct'] for r in eligible),
                              'errors': sum(r['error'] is not None for r in chosen),
                              'by_type': {t: {'n': sum(r['type'] == t for r in eligible),
                                              'correct': sum(r['correct'] and r['type'] == t for r in eligible)} for t in types},
                              'write_calls': len(write),
                              'write_input_tokens': sum(w['generation']['input_tokens'] for w in write),
                              'write_output_tokens': sum(w['generation'].get('output_tokens', 0) for w in write),
                              'write_seconds': sum(w['write_seconds'] for w in write),
                              'answer_input_tokens': sum(r['generation'].get('input_tokens') or 0 for r in chosen),
                              'answer_output_tokens': sum(r['generation'].get('output_tokens') or 0 for r in chosen),
                              'read_answer_seconds': sum(r['read_answer_seconds'] for r in chosen),
                              'unknown_failure_token_counts': sum(r['error'] is not None for r in chosen),
                              'output_cap_hits': sum(r['generation'].get('hit_output_cap_without_eos', False) for r in chosen)}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['write', 'answer'])
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--writes', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path; prior observations are preserved')
    protocol = json.loads(PROTOCOL.read_text())
    start = time.perf_counter()
    generator = LocalGenerator(args.model_dir, protocol)
    if args.phase == 'write':
        record = write_phase(protocol, args.model_dir, generator)
    else:
        if not args.writes:
            parser.error('--writes is required for answer phase')
        record = answer_phase(protocol, args.model_dir, generator, args.writes)
    record['phase_wall_seconds'] = time.perf_counter() - start
    args.output.write_text(json.dumps(record, indent=2) + '\n')
