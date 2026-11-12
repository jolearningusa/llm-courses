import copy
import unittest
from unittest.mock import patch
import numpy as np
import heldout_study as H


class StudyControls(unittest.TestCase):
    def test_mean_selection_failure_and_incomplete_ledgers(self):
        p = {'seeds': [1, 2], 'lr_grids': {'sgd': [0.1, 0.2]}}
        rows = [{'optimizer': 'sgd', 'lr': lr, 'seed': seed, 'status': 'finite',
                 'validation_mse': value, 'train_mse': -1000 if lr == 0.1 else 1000}
                for lr, values in [(0.1, [0.01, 9.0]), (0.2, [2.0, 2.0])]
                for seed, value in zip([1, 2], values)]
        self.assertEqual(H.select(rows, p)['sgd']['lr'], 0.2)
        rows[1]['validation_mse'] = .02
        rows[0]['status'] = 'nonfinite'; rows[0]['validation_mse'] = None
        self.assertEqual(H.select(rows, p)['sgd']['lr'], 0.2)
        with self.assertRaises(ValueError): H.select(rows[:-1], p)
        with self.assertRaises(ValueError): H.select(rows + rows[:1], p)

    def test_failed_trial_is_retained_and_budget_is_not_reallocated(self):
        p = {'data': {}, 'seeds': [1], 'steps': 2, 'batch': 1,
             'initialization_std': 0., 'lr_grids': {'sgd': [.1, 1e308]}}
        X, y = np.array([[1.], [2.], [3.]]), np.array([2., 4., 6.])
        ids = [np.array([0]), np.array([1]), np.array([2])]
        with patch.object(H, 'dataset', return_value=(X, y, ids)): report = H.search(p)
        self.assertEqual(len(report['trials']), 2)
        bad = report['trials'][1]
        self.assertEqual(bad['status'], 'nonfinite')
        self.assertEqual((bad['updates_attempted'], bad['updates_completed']), (1, 0))
        self.assertEqual(report['selected']['sgd']['lr'], .1)

    def test_id_and_feature_leak_rejected(self):
        X = np.array([[1., 2.], [3., 4.], [1., 2.]])
        with self.assertRaisesRegex(ValueError, 'overlapping'): H.validate_splits(X, [[0], [1], [0]])
        with self.assertRaisesRegex(ValueError, 'feature'): H.validate_splits(X, [[0], [1], [2]])

    def test_literal_final_update_and_heldout_independence(self):
        p = {'data': {}, 'seeds': [1], 'steps': 1, 'batch': 1, 'initialization_std': 0., 'lr_grids': {'sgd': [0.1]}}
        X = np.array([[1.], [2.], [3.]])
        y = np.array([2., 4., -999.])
        ids = [np.array([0]), np.array([1]), np.array([2])]
        with patch.object(H, 'dataset', return_value=(X, y, ids)):
            result = H.search(p)
            self.assertAlmostEqual(result['trials'][0]['weights'][0], .4)
            self.assertAlmostEqual(result['trials'][0]['validation_mse'], (4. - .8) ** 2)
            other_y = y.copy(); other_y[2] = 999.
            with patch.object(H, 'dataset', return_value=(X, other_y, ids)): other = H.search(p)
            self.assertEqual(result['selected'], other['selected'])
            self.assertEqual(result['trials'][0]['weights'], other['trials'][0]['weights'])
            tested = H.evaluate(p, result)
            self.assertAlmostEqual(tested['heldout'][0]['test_mse'], (1.2 + 999.) ** 2)
            self.assertEqual(tested['test_metric_calls'], 1)
            bad = copy.deepcopy(result); bad['selected']['sgd']['lr'] = .5
            with self.assertRaises(ValueError): H.evaluate(p, bad)
            bad = copy.deepcopy(result); bad['trials'][0]['weights'][0] += 1
            with self.assertRaises(ValueError): H.evaluate(p, bad)


if __name__ == '__main__': unittest.main()
