import unittest
from parallelism_lab import plan
class LayoutTests(unittest.TestCase):
    def test_layout_and_batch(self):
        a=plan(8,2,2,2,1,10**9,16,2e9,1e9,8,2)
        b=plan(8,2,1,4,1,10**9,16,2e9,1e9,8,2)
        self.assertEqual(a['estimated_peak_bytes_per_rank'],5e9)
        self.assertEqual(a['global_batch'],b['global_batch'])
        self.assertGreater(a['ideal_fill_drain_utilization'],b['ideal_fill_drain_utilization'])
    def test_context_replicates_state(self):
        a=plan(8,2,1,2,2,10**9,16,0,0,8,2)
        self.assertEqual(a['persistent_bytes_per_rank'],4e9)
    def test_invalid_mesh(self):
        with self.assertRaises(ValueError):plan(8,2,2,3,1,1,16,0,0,8,2)
