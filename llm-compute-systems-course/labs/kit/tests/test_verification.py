import math
import unittest
from verification_lab import (check_numerical, compare, fixture_traces,
                              reference, reordered, report, validate_trace,
                              wrong_epsilon, wrong_tail, zeros)


class VerificationTests(unittest.TestCase):
    def test_reference_and_legal_reorder(self):
        self.assertTrue(check_numerical(reference)['accepted_on_fixtures'])
        self.assertTrue(check_numerical(reordered)['accepted_on_fixtures'])

    def test_wrong_candidates_are_rejected(self):
        for fn in [zeros, wrong_tail, wrong_epsilon]:
            with self.subTest(fn=fn.__name__):
                self.assertFalse(check_numerical(fn)['accepted_on_fixtures'])

    def test_constant_zero_on_zero_alone_is_uninformative(self):
        self.assertTrue(compare(reference([0., 0.]), zeros([0., 0.])))
        self.assertFalse(compare(reference([1., 2.]), zeros([1., 2.])))

    def test_numerical_contract(self):
        self.assertFalse(compare([1.], [math.nan]))
        self.assertFalse(compare([1.], [1., 2.]))
        with self.assertRaises(ValueError):
            compare([1.], [1.], rtol=-1.)
        with self.assertRaises(ValueError):
            reference([])

    def test_ownership_and_generation_controls(self):
        expected = {'complete_then_release': True, 'queued_cancel': True,
                    'early_release': False, 'old_completion_after_reset': False,
                    'reset_drains_ownership': True}
        for name, trace in fixture_traces().items():
            self.assertEqual(validate_trace(trace)['accepted_on_trace'], expected[name], name)

    def test_duplicate_completion_and_leak(self):
        good = fixture_traces()['complete_then_release']
        self.assertFalse(validate_trace(good[:-1])['accepted_on_trace'])
        self.assertFalse(validate_trace(good[:3]+[good[2]]+[good[3]])['accepted_on_trace'])

    def test_unknown_and_inflight_cancel(self):
        trace = fixture_traces()['complete_then_release'][:2]
        self.assertFalse(validate_trace(trace+[{'op': 'cancel', 'cmd': 'c0', 'generation': 0}])['accepted_on_trace'])
        self.assertFalse(validate_trace([{'op': 'unknown', 'cmd': 'x', 'generation': 0}])['accepted_on_trace'])

    def test_report_is_not_a_performance_claim(self):
        data = report()
        self.assertIn('CPU', data['provenance'])
        self.assertNotIn('speedup', data)
        self.assertEqual(len(data['numerical']), 5)


if __name__ == '__main__':
    unittest.main()
