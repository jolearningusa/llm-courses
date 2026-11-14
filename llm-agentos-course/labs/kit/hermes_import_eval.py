"""Probe supplied Hermes interfaces; no model, compactor or provider service runs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import tempfile

KIT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source, revision, output):
    source = source.resolve()
    if output.exists():
        raise FileExistsError('Choose a new result path')
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Use a fresh isolated venv')
    ledger = json.loads((KIT / '../../references/hermes_import_sources_2026_11.json').resolve().read_text())
    cohort = next(row for row in ledger['cohorts'] if row['revision'] == revision)
    for name, expected in cohort['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('Upstream source differs from the pinned cohort: ' + name)
    sys.path[:0] = [str(source), str(KIT)]
    import agent.context_engine as context
    import agent.memory_provider as memory
    from nanoos.context import ContextEngine as ViewEngine
    from nanoos import ports

    assert ports.HermesContextEngine.__mro__[1] is context.ContextEngine
    assert ports.HermesMemoryProvider.__mro__[1] is memory.MemoryProvider
    assert not inspect.isabstract(ports.HermesContextEngine)
    assert not inspect.isabstract(ports.HermesMemoryProvider)
    parent = ports.HermesContextEngine(lambda budget: ViewEngine(), context_length=1000)
    parent.update_from_response({'prompt_tokens': 600, 'completion_tokens': 4})
    child = parent.clone_for_agent()
    assert isinstance(child, context.ContextEngine)
    assert child is not parent and child.engine is not parent.engine
    child.update_model('fixture-small', 200)
    child.update_from_response({'prompt_tokens': 12, 'completion_tokens': 3})
    assert (parent.context_length, parent.threshold_tokens, parent.last_total_tokens) == (1000, 750, 604)
    assert (child.context_length, child.threshold_tokens, child.last_total_tokens) == (200, 150, 15)
    provider = ports.HermesMemoryProvider(lambda: None)
    assert isinstance(provider, memory.MemoryProvider)
    assert provider.is_available() and provider.get_tool_schemas()[0]['name'] == 'nanoos_memory_search'
    # The supplied scaffold is intentionally not represented as an implemented compactor.
    try:
        parent.compress([{'role': 'user', 'content': 'fixture'}])
    except NotImplementedError:
        compaction = 'student TODO; not qualified'
    else:
        raise AssertionError('Scaffold changed: review compaction scope before recording a new cohort')
    imported = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).resolve().is_relative_to(source):
            relative = str(Path(filename).resolve().relative_to(source))
            imported[name] = {'file': relative, 'sha256': digest(Path(filename))}
            assert relative in cohort['source_sha256'], 'Unrecorded upstream import: ' + relative
    jiter = sys.modules.get('agent.jiter_preload')
    record = {'revision': revision, 'version': cohort['version'], 'python': sys.version,
              'platform': platform.platform(), 'isolated_venv': sys.prefix != sys.base_prefix,
              'venv_config': (Path(sys.prefix) / 'pyvenv.cfg').read_text(),
              'installed_distributions': sorted((d.metadata['Name'], d.version) for d in __import__('importlib.metadata', fromlist=['distributions']).distributions()),
              'source_archive_sha256': cohort['archive_sha256'], 'imported_upstream': imported,
              'adapter_sha256': digest(KIT / 'nanoos/ports.py'),
              'probe_sha256': digest(Path(__file__)),
              'checks': {'actual_context_base': True, 'actual_memory_base': True,
                         'abstract_contract_constructs': True, 'fresh_adapter_clone': True,
                         'child_budget_usage_isolation': True, 'provider_discovery_schema': True},
              'counters': {'parent': parent.get_status(), 'child': child.get_status()},
              'signatures': {name: str(inspect.signature(getattr(context.ContextEngine, name)))
                             for name in ('compress', 'select_context', 'clone_for_agent')},
              'optional_jiter_preload': {'loaded': getattr(jiter, '_JITER_PRELOADED', None),
                                        'error_type': type(getattr(jiter, '_JITER_PRELOAD_ERROR', None)).__name__},
              'compaction': compaction, 'session_retry': 'not run', 'full_hermes_installation': False,
              'passed': True}
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(cohort['version'], 'six import/construction/clone checks passed; compaction/retry not qualified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='hermes-contract-home-') as home:
        import os
        os.environ['HERMES_HOME'] = home
        evaluate(args.source_root, args.revision, args.output)
