"""Two-stage, training-only search then locked held-out evaluation; NumPy CPU."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import optimizers as O
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'L0_linear_logistic'))
import linear_models as L


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources():
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(), HERE / 'optimizers.py', Path(L.__file__)]}


def validate_splits(X, ids):
    groups = [set(map(int, group)) for group in ids]
    if any(len(group) != len(original) for group, original in zip(groups, ids)):
        raise ValueError('duplicate sample ID')
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
        raise ValueError('overlapping split IDs')
    rows = [{X[i].tobytes() for i in group} for group in groups]
    if any(a & b for i, a in enumerate(rows) for b in rows[i + 1:]):
        raise ValueError('exact feature row shared across splits')


def dataset(p):
    d = p['data']
    X, y, _ = O.noisy_regression(n=d['n'], d=d['d'], noise=d['noise'], seed=d['pool_seed'])
    ids = np.split(np.random.default_rng(d['split_seed']).permutation(len(y)), np.cumsum(d['counts'])[:-1])
    validate_splits(X, ids)
    return X, y, ids


def select(rows, p):
    """Only validation decides; require every planned seed and LR exactly once."""
    expected = {(name, float(lr), seed) for name, grid in p['lr_grids'].items() for lr in grid for seed in p['seeds']}
    keys = [(row['optimizer'], row['lr'], row['seed']) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('incomplete or duplicate trial ledger')
    chosen = {}
    for name, grid in p['lr_grids'].items():
        candidates = []
        for order, lr in enumerate(grid):
            trials = [row for row in rows if row['optimizer'] == name and row['lr'] == lr]
            if all(row['status'] == 'finite' and row['validation_mse'] is not None and np.isfinite(row['validation_mse']) for row in trials):
                candidates.append((sum(row['validation_mse'] for row in trials) / len(trials), order, lr))
        if not candidates: raise ValueError('no eligible candidate for ' + name)
        mean, _, lr = min(candidates)
        chosen[name] = {'lr': lr, 'mean_validation_mse': mean}
    return chosen


def search(p):
    start = time.perf_counter()
    X, y, ids = dataset(p)
    train, val, _ = ids
    rows = []
    for name, grid in p['lr_grids'].items():
        for lr in grid:
            for seed in p['seeds']:
                rng = np.random.default_rng(seed)
                w = rng.normal(scale=p['initialization_std'], size=X.shape[1])
                init_hash = digest(w.tolist())
                batches = rng.integers(0, len(train), size=(p['steps'], p['batch']))
                fixed = p.get('fixed', {})
                kw = ({'b1': fixed.get('adam_b1', .9), 'b2': fixed.get('adam_b2', .999),
                       'eps': fixed.get('adam_eps', 1e-8), 'weight_decay': fixed.get('weight_decay', 0.)}
                      if name == 'adam' else {})
                opt = O.make(name, **kw)
                row = {'optimizer': name, 'lr': lr, 'seed': seed, 'status': 'finite',
                       'initial_weights_sha256': init_hash, 'batch_indices_sha256': digest(batches.tolist()),
                       'updates_attempted': 0, 'updates_completed': 0, 'validation_mse': None}
                before = time.perf_counter()
                try:
                    with np.errstate(over='raise', invalid='raise', divide='raise'):
                        for batch in batches:
                            idx = train[batch]
                            grad = L.mse_grad(X[idx], y[idx], w)
                            row['updates_attempted'] += 1
                            opt.step([w], [grad], lr)
                            if not np.all(np.isfinite(w)): raise FloatingPointError('nonfinite weights')
                            row['updates_completed'] += 1
                        row['train_mse'] = L.mse(X[train], y[train], w)
                        row['validation_mse'] = L.mse(X[val], y[val], w)
                        if not np.isfinite(row['validation_mse']): raise FloatingPointError('nonfinite validation MSE')
                        row['weights'] = w.tolist()
                        row['weights_sha256'] = digest(row['weights'])
                except FloatingPointError as exc:
                    row.update(status='nonfinite', validation_mse=None, error=str(exc))
                row['fit_and_validation_seconds'] = time.perf_counter() - before
                rows.append(row)
    result = {'protocol_sha256': digest(p), 'sources': sources(),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
              'split_ids': {name: group.tolist() for name, group in zip(['train', 'validation', 'test'], ids)},
              'data_sha256': hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
              'trials': rows, 'selected': select(rows, p), 'test_metric_calls': 0}
    result['selection_sha256'] = digest(result['selected'])
    result['search_seconds'] = time.perf_counter() - start
    return result


def evaluate(p, locked):
    """Receives a saved selection; no fitting or candidate extension occurs here."""
    if locked['protocol_sha256'] != digest(p) or locked['sources'] != sources(): raise ValueError('protocol or code changed after search')
    if locked['selected'] != select(locked['trials'], p) or locked['selection_sha256'] != digest(locked['selected']): raise ValueError('selection changed after validation')
    X, y, ids = dataset(p)
    if locked['data_sha256'] != hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(): raise ValueError('data changed')
    test = ids[2]
    results = []
    start = time.perf_counter()
    for row in locked['trials']:
        if row['lr'] == locked['selected'][row['optimizer']]['lr']:
            if digest(row['weights']) != row['weights_sha256']: raise ValueError('checkpoint changed')
            error = X[test] @ np.asarray(row['weights']) - y[test]
            results.append({'optimizer': row['optimizer'], 'lr': row['lr'], 'seed': row['seed'],
                            'weights_sha256': row['weights_sha256'], 'test_mse': float(np.mean(error ** 2))})
    return {'protocol_sha256': digest(p), 'search_artifact_sha256': digest(locked),
            'selection_sha256': locked['selection_sha256'], 'heldout': results,
            'test_metric_calls': len(results), 'additional_fit_updates': 0, 'test_seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['search', 'evaluate'])
    parser.add_argument('--protocol', type=Path, default=HERE / 'heldout_protocol.json')
    parser.add_argument('--search', type=Path, default=HERE / 'heldout_search_report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists; preserve the original study artifact')
    p = json.loads(args.protocol.read_text())
    result = search(p) if args.stage == 'search' else evaluate(p, json.loads(args.search.read_text()))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__': main()
