"""Test optional-Hermes import boundaries in fresh processes, without an endpoint."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

KIT = Path(__file__).resolve().parents[1]


class TestHermesImports(unittest.TestCase):
    def probe(self, context=None, memory='class MemoryProvider: pass', helper=None):
        with tempfile.TemporaryDirectory(prefix='hermes-import-fixture-') as folder:
            root = Path(folder)
            if context is not None:
                agent = root / 'agent'
                agent.mkdir()
                (agent / '__init__.py').write_text('')
                (agent / 'context_engine.py').write_text(context)
                (agent / 'memory_provider.py').write_text(memory)
                if helper is not None:
                    (agent / 'helper.py').write_text(helper)
            code = ('import sys,json\n'
                    # Exclude any installed agent package from this synthetic boundary fixture.
                    'sys.path=[p for p in sys.path if "site-packages" not in p]\n'
                    f'sys.path[:0]=[{str(root)!r},{str(KIT)!r}]\n'
                    'from nanoos import ports\n'
                    'print(json.dumps({"base":ports.HermesContextEngine.__mro__[1].__module__}))\n')
            return subprocess.run([sys.executable, '-I', '-c', code],
                                  cwd=root, capture_output=True, text=True, timeout=15)

    def test_absent_hermes_allows_offline_standins(self):
        p = self.probe()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('"base": "nanoos.ports"', p.stdout)

    def test_missing_nested_dependency_is_not_hidden(self):
        p = self.probe('import absent_hermes_contract_dependency\nclass ContextEngine: pass')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('absent_hermes_contract_dependency', p.stderr)

    def test_partial_agent_package_is_not_hidden(self):
        # Existing agent package with a context module that cannot find another module.
        p = self.probe('from agent.absent_helper import ContextEngine')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('agent.absent_helper', p.stderr)

    def test_removed_symbol_is_not_hidden(self):
        p = self.probe('from agent.helper import RemovedSymbol', helper='available = 1')
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('RemovedSymbol', p.stderr)

    def test_present_interfaces_are_used(self):
        p = self.probe('class ContextEngine: pass')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('"base": "agent.context_engine"', p.stdout)


if __name__ == '__main__':
    unittest.main()
