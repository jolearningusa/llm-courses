import unittest
from gradient_diagnostics import (autograd_cubic, centered_difference, cubic,
                                  error_metrics, matches, report)


class GradientDiagnostics(unittest.TestCase):
    def test_smooth_autograd_matches_closed_form_and_difference(self):
        for x in (-0.7, 0.2, 1.0):
            self.assertAlmostEqual(autograd_cubic(x), 3*x*x, places=12)
            self.assertTrue(matches(3*x*x, centered_difference(cubic, x, 1e-5)))

    def test_accepted_control_and_rejected_mutants(self):
        self.assertEqual({r['name']: r['accepted_on_fixture'] for r in report()['controls']},
                         {'autograd': True, 'wrong_sign': False, 'missing_contribution': False})

    def test_tiny_step_can_destroy_a_correct_check(self):
        self.assertEqual(centered_difference(cubic, 1.0, 1e-20), 0.0)
        self.assertFalse(matches(3.0, centered_difference(cubic, 1.0, 1e-20)))

    def test_kink_is_not_a_smooth_oracle(self):
        wide, narrow, at_zero = report()['relu']
        self.assertTrue(wide['touches_kink'])
        self.assertAlmostEqual(wide['numeric'], 0.45)
        self.assertFalse(narrow['touches_kink'])
        self.assertEqual(narrow['numeric'], 0.0)
        self.assertIsNone(at_zero['classical_derivative'])
        self.assertAlmostEqual(at_zero['numeric'], 0.5)

    def test_near_zero_needs_an_absolute_scale(self):
        self.assertEqual(error_metrics(0., 0.), {'absolute': 0., 'raw_relative': 0.})
        errors = error_metrics(0., 1e-12)
        self.assertEqual(errors['raw_relative'], 1.0)
        self.assertEqual(errors['absolute'], 1e-12)
        self.assertTrue(matches(0., 1e-12))
        self.assertFalse(matches(0., 1e-12, atol=0.))

    def test_invalid_or_nonfinite_contract_is_not_a_pass(self):
        self.assertFalse(matches(1., float('nan')))
        self.assertFalse(matches(float('inf'), float('inf')))
        with self.assertRaises(ValueError): matches(1., 1., atol=-1.)
        for h in (0., -1., float('inf'), float('nan')):
            with self.assertRaises(ValueError): centered_difference(cubic, 1., h)


if __name__ == '__main__':
    unittest.main()
