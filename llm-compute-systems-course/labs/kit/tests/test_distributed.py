import unittest
from distributed_lab import ring_seconds
class CollectiveModelTests(unittest.TestCase):
    def test_worked_example(self):
        self.assertAlmostEqual(ring_seconds(8,256e6,5e-6,50e9),0.00903)
        self.assertEqual(ring_seconds(1,256e6,5e-6,50e9),0)
    def test_invalid(self):
        for args in [(0,1,1,1),(2,-1,1,1),(2,1,-1,1),(2,1,1,0)]:
            with self.assertRaises(ValueError):ring_seconds(*args)
