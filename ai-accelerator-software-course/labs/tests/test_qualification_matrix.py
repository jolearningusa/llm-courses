"""Policy controls use fabricated records; none represents device execution."""
import copy
import unittest

from qualification_matrix import decide, report


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.identity = dict(source='source-a', build='build-a', target='board-a', run='run-a')
        self.row = dict(stage='device_run', identity=self.identity,
                        checks={'functional': dict(status='passed', evidence=['harness-log'])})

    def decision(self, row=None, **kwargs):
        return decide(self.row if row is None else row, self.identity,
                      ['functional'], 'physical', **kwargs)

    def test_complete_fabricated_record_is_policy_positive_control(self):
        self.assertTrue(self.decision()['accepted'])

    def test_unavailable_board_cannot_pass_even_with_claimed_device_run(self):
        self.assertFalse(self.decision(available=False)['accepted'])

    def test_build_and_simulator_are_not_physical_evidence(self):
        for stage in ('built_only', 'simulator_run'):
            with self.subTest(stage=stage):
                self.row['stage'] = stage
                self.assertFalse(self.decision()['accepted'])

    def test_failed_and_missing_evidence_remain_visible_together(self):
        self.row['checks']['functional'] = dict(status='failed')
        result = self.decision()
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['known_failures'], ['functional'])
        self.row['checks']['functional']['evidence'] = ['failure-log']
        self.assertEqual(self.decision()['status'], 'failed')

    def test_unsupported_and_not_run_are_incomplete_not_passed(self):
        for status in ('unsupported', 'not_run'):
            with self.subTest(status=status):
                self.row['checks']['functional'] = dict(status=status, reason='fixture limitation')
                self.assertEqual(self.decision()['status'], 'incomplete')
                self.row['checks']['functional'].pop('reason')
                self.assertEqual(self.decision()['status'], 'invalid')

    def test_identity_mismatch_or_absence_blocks_acceptance(self):
        for field in self.identity:
            row = copy.deepcopy(self.row)
            row['identity'][field] = 'other'
            self.assertEqual(self.decision(row)['status'], 'invalid')
            row['identity'][field] = None
            self.assertFalse(self.decision(row)['accepted'])

    def test_missing_extra_or_malformed_required_checks(self):
        for checks in ({}, {'functional': None}, {'functional': {'status': 'clean'}},
                       dict(self.row['checks'], extra={'status': 'passed'}), []):
            self.row['checks'] = checks
            self.assertEqual(self.decision()['status'], 'invalid')

    def test_unknown_stage_or_scope_and_empty_policy(self):
        self.row['stage'] = 'done'
        self.assertEqual(self.decision()['status'], 'invalid')
        for scope, required in [('anything', ['functional']), ('physical', []),
                                ('physical', ['functional', 'functional'])]:
            with self.assertRaises(ValueError):
                decide(self.row, self.identity, required, scope)

    def test_executed_cpu_report_has_no_physical_pass(self):
        data = report()
        self.assertEqual(data['decisions']['build']['status'], 'incomplete')
        self.assertEqual(data['decisions']['model']['status'], 'passed')
        self.assertFalse(data['decisions']['physical']['accepted'])
        self.assertEqual(data['rows']['physical']['stage'], 'not_run')
        self.assertTrue(all(x['status'] == 'not_run'
                            for x in data['rows']['physical']['checks'].values()))
        self.assertEqual(len(data['bytecode_sha256']), 3)


if __name__ == '__main__':
    unittest.main()
