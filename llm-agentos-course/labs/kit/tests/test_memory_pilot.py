"""Independent controls for the pilot's isolation, eligibility and literal scorer."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'memory_pilot.py'
spec = importlib.util.spec_from_file_location('memory_pilot', PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
P = json.loads(pilot.PROTOCOL.read_text())


class TestMemoryPilot(unittest.TestCase):
    def test_literal_scorer_rejects_negation_and_extra_prose(self):
        self.assertTrue(pilot.score('  GHENT. ', 'Ghent'))
        self.assertFalse(pilot.score('not Ghent', 'Ghent'))
        self.assertFalse(pilot.score('Ghent or Porto', 'Ghent'))
        self.assertFalse(pilot.score('', 'Ghent'))

    def test_retrieval_ranking_and_stable_ties(self):
        chunks = ['red object', 'red notebook', 'notebook red', 'green mug']
        selected = pilot.retrieve(chunks, 'red notebook', P)
        self.assertEqual([r['index'] for r in selected], [1, 2])
        self.assertEqual([r['score'] for r in selected], [2, 2])

    def test_full_is_complete_and_none_has_no_history(self):
        history = {'id': 'alice', 'turns': ['first', 'middle', 'last']}
        self.assertEqual(pilot.context_for('full', history, 'q', {}, P)[0], 'first\nmiddle\nlast')
        self.assertEqual(pilot.context_for('none', history, 'q', {}, P)[0], '[none]')

    def test_profile_scope_and_query_only_prompt_construction(self):
        h = {'id': 'alice', 'turns': ['alice likes tea']}
        stores = {'alice': {'chunks': ['alice likes tea']}, 'bob': {'chunks': ['bob secret 987']}}
        messages, selected = pilot.answer_messages('extracted', h, 'What does alice like?', stores, P)
        self.assertNotIn('987', json.dumps(messages))
        self.assertEqual(selected[0]['text'], 'alice likes tea')
        changed = copy.deepcopy(stores)
        changed['bob']['chunks'] = ['unrelated replacement']
        self.assertEqual(pilot.answer_messages('extracted', h, 'What does alice like?', changed, P), (messages, selected))

    def test_over_cap_is_ineligible_without_generation(self):
        class Tokenizer:
            eos_token_id = 0
            def apply_chat_template(self, *a, **kw): return 'long prompt'
            def __call__(self, *a, **kw):
                return {'input_ids': type('IDs', (), {'shape': (1, 513)})()}
        obj = pilot.LocalGenerator.__new__(pilot.LocalGenerator)
        obj.tokenizer, obj.protocol = Tokenizer(), P
        obj.model = None  # A generation attempt would fail; it must never be reached.
        row = obj.generate([], 16)
        self.assertFalse(row['eligible'])
        self.assertEqual((row['input_tokens'], row['reason']), (513, 'input_cap'))

    def test_extraction_does_not_read_questions_or_labels(self):
        class ForbiddenQuestions:
            def read_text(self): raise AssertionError('Writer read questions')
        class Generator:
            def generate(self, messages, limit):
                self.messages = messages
                return {'eligible': True, 'text': 'one fact', 'input_tokens': 5, 'output_tokens': 2}
        with tempfile.TemporaryDirectory() as folder:
            h = Path(folder) / 'histories.json'
            h.write_text(json.dumps([{'id': 'x', 'turns': ['Only this history is visible.']}]))
            gen = Generator()
            with patch.object(pilot, 'HISTORIES', h), patch.object(pilot, 'QUESTIONS', ForbiddenQuestions()), patch.object(pilot, 'base_record', return_value={}):
                result = pilot.write_phase(P, Path(folder), gen)
            self.assertEqual(gen.messages[1]['content'], 'Only this history is visible.')
            self.assertEqual(result['writes'][0]['chunks'], ['one fact'])

    def test_write_store_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            record = {'protocol_sha256': pilot.sha(pilot.PROTOCOL), 'histories_sha256': pilot.sha(pilot.HISTORIES),
                      'model_files_sha256': {}, 'model_revision': P['model_revision'],
                      'runner_sha256': pilot.sha(PATH),
                      'writes': [{'history_id': 'x', 'chunks': ['fact'], 'store_sha256': pilot.object_sha(['fact'])}]}
            self.assertIn('x', pilot.verify_writes(record, P, model))
            record['writes'][0]['chunks'] = ['changed fact']
            with self.assertRaisesRegex(ValueError, 'store mutation'):
                pilot.verify_writes(record, P, model)

    def test_summary_keeps_failed_and_ineligible_planned_cases(self):
        rows = []
        for condition in P['conditions']:
            for i in range(2):
                rows.append({'condition': condition, 'type': 'fact', 'correct': i == 0,
                             'error': 'controlled failure' if i == 1 else None,
                             'read_answer_seconds': 1.0,
                             'generation': {'eligible': True, 'input_tokens': 2 if i == 0 else None,
                                            'output_tokens': 1 if i == 0 else None}})
        rows[1]['generation']['eligible'] = False
        summary = pilot.summarize(rows, {'writes': []}, P)
        self.assertEqual((summary['none']['planned'], summary['none']['eligible'], summary['none']['correct']), (2, 1, 1))
        self.assertEqual((summary['full']['planned'], summary['full']['eligible'], summary['full']['correct']), (2, 2, 1))
        self.assertEqual(summary['full']['errors'], 1)
        self.assertEqual(summary['full']['unknown_failure_token_counts'], 1)


if __name__ == '__main__':
    unittest.main()
