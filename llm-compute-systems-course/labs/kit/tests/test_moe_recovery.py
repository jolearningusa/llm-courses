import unittest
try:
    import torch
except ImportError:
    torch=None
from moe_lab import routed,token_reference,fixture
from recovery_lab import experiment

@unittest.skipIf(torch is None,'pinned PyTorch environment required')
class MoeRecoveryTests(unittest.TestCase):
    def test_skew_empty_experts_and_combine(self):
        for skew in (False,True):
            args=fixture(skew)
            actual,counts=routed(*args)
            torch.testing.assert_close(actual,token_reference(*args),rtol=1e-5,atol=1e-5)
            self.assertEqual(sum(counts),34)
            if skew:self.assertEqual(counts,[17,17,0,0])
    def test_invalid_route(self):
        args=list(fixture())
        args[3][0,0]=4
        with self.assertRaises(ValueError):routed(*args)
    def test_resume_and_rng_negative_control(self):
        good=experiment(True)
        self.assertEqual(good['parameter_max_abs_error'],0)
        self.assertEqual(good['optimizer_max_abs_error'],0)
        self.assertEqual(good['baseline_losses'],good['resumed_losses'])
        self.assertGreater(experiment(False)['parameter_max_abs_error'],0)
