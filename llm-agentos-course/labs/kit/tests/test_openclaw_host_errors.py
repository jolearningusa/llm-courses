"""Provided host regression: do not conceal the first failed compaction."""
import unittest
from models.scripted import ScriptedModel
from nanoos.hosts import OpenClawHost


class NoTools:
    def schemas(self):
        return []


class ContextFixture:
    def __init__(self, *, assemble_error=None, compact_error=None):
        self.assemble_error = assemble_error
        self.compact_error = compact_error
        self.compact_calls = 0
        self.assemble_calls = 0
        self.compacted = False

    def call(self, method, params=None):
        if method == 'info':
            return {'ownsCompaction': True}
        if method == 'assemble':
            self.assemble_calls += 1
            if self.assemble_error:
                raise self.assemble_error
            return {'messages': [{'role': 'user', 'content': 'short' if self.compacted else 'x' * 4000}]}
        if method == 'compact':
            self.compact_calls += 1
            if self.compact_error:
                raise self.compact_error
            self.compacted = True
            return {'ok': True, 'compacted': True}
        return {'ingested': True}


class OpenClawHostErrorTests(unittest.TestCase):
    def host(self, engine):
        model = ScriptedModel([{'content': 'known-good reply'}])
        return OpenClawHost(model, NoTools(), engine, token_budget=64, system_prompt='sys'), model

    def test_first_compaction_error_visible_before_model_then_quarantined_retry(self):
        sentinel = RuntimeError('deliberate compaction failure')
        engine = ContextFixture(compact_error=sentinel)
        host, model = self.host(engine)
        with self.assertRaises(RuntimeError) as caught:
            host.run(['first attempt'])
        self.assertIs(caught.exception, sentinel)
        self.assertEqual(model.calls, [], 'failed compaction must not look like a successful reply')
        self.assertEqual(host.quarantined, 'compact: deliberate compaction failure')
        self.assertEqual(engine.compact_calls, 1)
        result = host.run(['explicit later attempt'])
        self.assertEqual(result.answers, ['known-good reply'])
        self.assertEqual(result.compressions, 0)
        self.assertEqual(engine.compact_calls, 1, 'quarantined engine must not be called again')
        self.assertEqual(len(model.calls), 1)
        self.assertLessEqual(result.prompt_tokens[0], 64)

    def test_assembly_error_still_falls_back_on_same_attempt(self):
        engine = ContextFixture(assemble_error=ValueError('deliberate assembly failure'))
        host, model = self.host(engine)
        result = host.run(['task'])
        self.assertEqual(result.answers, ['known-good reply'])
        self.assertEqual(host.quarantined, 'assemble: deliberate assembly failure')
        self.assertEqual(engine.assemble_calls, 1)
        self.assertEqual(engine.compact_calls, 0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.compressions, 0)

    def test_successful_compaction_keeps_normal_reply_and_accounting(self):
        engine = ContextFixture()
        host, model = self.host(engine)
        result = host.run(['task'])
        self.assertEqual(result.answers, ['known-good reply'])
        self.assertIsNone(host.quarantined)
        self.assertEqual(engine.compact_calls, 1)
        self.assertEqual(engine.assemble_calls, 2)
        self.assertEqual(result.compressions, 1)
        self.assertIn('compact', result.hooks)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.requests[0], [{'role': 'user', 'content': 'short'}])


if __name__ == '__main__':
    unittest.main()
