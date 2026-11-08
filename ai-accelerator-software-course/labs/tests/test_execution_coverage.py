import json
import unittest

from execution_coverage import TARGET, classify, run_control


class ExecutionCoverageTests(unittest.TestCase):
    def test_legal_transfer_has_actual_start_completion_and_output(self):
        row = run_control('legal')
        self.assertEqual(row['output'], [0, 1, 254, 255])
        self.assertTrue(row['decision']['accepted'])
        self.assertEqual(json.loads(row['coverage_channel'])['started'], 1)
        self.assertEqual([e['event'] for e in row['trace']][-4:],
                         ['start', 'output_visible', 'completion', 'reap'])

    def test_no_launch_cannot_pass_with_right_numbers_and_fake_stdout(self):
        row = run_control('no_launch')
        self.assertEqual(row['decision']['correctness'], 'passed')
        self.assertEqual(row['decision']['status'], 'invalid')
        self.assertIn('1000', row['candidate_stdout_ignored'])
        self.assertFalse(any(e['event'] == 'start' for e in row['trace']))

    def test_missing_malformed_and_replayed_records_are_invalid(self):
        for name in ['missing_coverage', 'malformed_coverage', 'replayed_coverage']:
            with self.subTest(name=name):
                row = run_control(name)
                self.assertEqual(row['decision']['correctness'], 'passed')
                self.assertEqual(row['decision']['diagnostics'], 'passed')
                self.assertEqual(row['decision']['status'], 'invalid')

    def test_strict_schema_counts_and_target_identity(self):
        row = run_control('legal')
        cov = json.loads(row['coverage_channel'])
        for field, value in [('started', True), ('started', 1.0), ('started', 0),
                             ('started', -1), ('started', 2), ('completed', 0),
                             ('target', 'unrelated-work'), ('nonce', 'stale')]:
            with self.subTest(field=field, value=value):
                verdict = classify(json.dumps(dict(cov, **{field: value})),
                                   row['diagnostics_channel'], row['nonce'], True)
                self.assertEqual(verdict['status'], 'invalid')
        self.assertEqual(cov['target'], TARGET)
        duplicate = row['coverage_channel'].replace('"started": 1', '"started": 0, "started": 1')
        self.assertEqual(classify(duplicate, row['diagnostics_channel'], row['nonce'], True)['status'], 'invalid')

    def test_valid_coverage_does_not_excuse_wrong_output(self):
        row = run_control('wrong_result')
        self.assertEqual(row['decision']['coverage'], 'valid')
        self.assertEqual(row['decision']['diagnostics'], 'passed')
        self.assertEqual(row['decision']['status'], 'failed')

    def test_error_completion_is_not_a_clean_execution(self):
        row = run_control('dma_error')
        self.assertEqual(row['decision']['coverage'], 'valid')
        self.assertEqual(row['decision']['diagnostics'], 'failed')
        self.assertEqual(row['decision']['status'], 'failed')
        self.assertFalse(any(e['event'] == 'output_visible' for e in row['trace']))

    def test_bounds_rejected_before_launch_stays_visible(self):
        row = run_control('rejected_bounds')
        self.assertEqual(row['decision']['status'], 'invalid')
        self.assertEqual(row['decision']['diagnostics'], 'failed')
        self.assertFalse(any(e['event'] == 'submit' for e in row['trace']))
        self.assertIn('bounds violation', row['diagnostics_channel'])

    def test_missing_or_malformed_diagnostics_cannot_pass(self):
        row = run_control('missing_diagnostics')
        self.assertEqual(row['decision']['coverage'], 'valid')
        self.assertEqual(row['decision']['status'], 'invalid')
        legal = run_control('legal')
        for raw in ['{}', '[]', '{bad', json.dumps({'nonce': legal['nonce'],
                    'check': 'cpu-model-transfer-v1', 'findings': 'none'})]:
            self.assertEqual(classify(legal['coverage_channel'], raw, legal['nonce'], True)['status'], 'invalid')


if __name__ == '__main__':
    unittest.main()
