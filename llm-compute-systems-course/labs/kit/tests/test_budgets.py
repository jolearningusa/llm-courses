import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budgets import gemm, kv_bytes, training_state


class BudgetTests(unittest.TestCase):
    def test_gqa_worked_example_and_variable_lengths(self):
        self.assertEqual(kv_bytes(32,8,128,[32768]), 4*1024**3)
        self.assertEqual(kv_bytes(32,8,128,[100,0,200]), kv_bytes(32,8,128,[300]))
        self.assertEqual(kv_bytes(32,32,128,[32768]), 16*1024**3)
        self.assertEqual(kv_bytes(32,8,128,[]), 0)

    def test_gemm_units_and_regime_transition(self):
        large=gemm(4096,4096,4096)
        small=gemm(1,4096,4096)
        self.assertEqual(large['flops'],137438953472)
        self.assertEqual(large['ideal_traffic_bytes'],96*1024**2)
        self.assertEqual(large['limiting_term'],'compute')
        self.assertEqual(small['limiting_term'],'memory')
        self.assertAlmostEqual(large['math_bound_seconds'],.00137438953472)

    def test_training_stages_conserve_unsharded_case(self):
        for stage in range(4):
            self.assertEqual(training_state(100,1,stage)['persistent_bytes_per_rank'],1600)
        totals=[training_state(100,8,s)['persistent_bytes_per_rank'] for s in range(4)]
        self.assertEqual(totals,[1600,550,375,200])
        self.assertEqual(training_state(100,8,3,optimizer_bytes=8)['persistent_bytes_per_rank'],150)

    def test_reject_invalid_inputs(self):
        for value in [0,-1,float('nan'),float('inf'),True,1.5]:
            with self.subTest(value=value), self.assertRaises(ValueError):gemm(value,2,2)
        for value in [0,-1,float('nan'),float('inf')]:
            with self.subTest(rate=value), self.assertRaises(ValueError):gemm(2,2,2,compute_tflops=value)
        for lengths in [[-1],[1.5],[True]]:
            with self.assertRaises(ValueError):kv_bytes(2,2,2,lengths)
        with self.assertRaises(ValueError):training_state(100,stage=4)
        with self.assertRaises(ValueError):training_state(100,ranks=0)


if __name__=='__main__':unittest.main()
