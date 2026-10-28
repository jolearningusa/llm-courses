import unittest
from system_efficiency import amdahl,ledger
class EfficiencyTests(unittest.TestCase):
    def test_phase_ledger_matches_amdahl(self):
        p={'compute':60.,'other':40.};a=ledger(12000,p);b=ledger(12000,p,'compute',2.)
        self.assertEqual(a['elapsed_s'],100.)
        self.assertEqual(b['elapsed_s'],70.)
        self.assertAlmostEqual(b['useful_tokens_per_s']/a['useful_tokens_per_s'],amdahl(.6,2.))
        self.assertEqual(amdahl(0,20),1.)
    def test_reject_invalid_accounting(self):
        for args in [(-1,{'x':1}), (1,{'x':-1}), (1,{'x':0})]:
            with self.assertRaises(ValueError):ledger(*args)
        with self.assertRaises(ValueError):amdahl(1.1,2)
