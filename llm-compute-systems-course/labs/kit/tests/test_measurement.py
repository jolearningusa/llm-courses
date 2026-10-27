import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from measurement import summarize,interval_accounting,benchmark


class MeasurementTests(unittest.TestCase):
    def test_summary_and_percentile_convention(self):
        r=summarize([4,1,3,2])
        self.assertEqual(r['median_seconds'],2.5)
        self.assertEqual(r['p95_nearest_rank_seconds'],4)
        self.assertEqual(r['samples_seconds'],[4,1,3,2])

    def test_union_clipping_and_nested_overlap(self):
        r=interval_accounting([(-5,1),(0,4),(2,7),(8,15),(3,5)],0,10)
        self.assertEqual(r['busy_duration'],9)
        self.assertEqual(r['uncovered_duration'],1)
        self.assertEqual(r['merged_intervals'],[[0,7],[8,10]])
        self.assertEqual(interval_accounting([],0,10)['busy_duration'],0)

    def test_warmup_excluded_with_controlled_clock(self):
        ticks=iter([0,2,10,14]);calls=[]
        r=benchmark(lambda:calls.append(1) or len(calls),repeats=2,warmup=3,clock=lambda:next(ticks))
        self.assertEqual(len(calls),5)
        self.assertEqual(r['samples_seconds'],[2,4])
        self.assertEqual(r['result'],5)

    def test_invalid_data(self):
        for xs in [[],[-1],[float('nan')],[float('inf')],[True]]:
            with self.assertRaises(ValueError):summarize(xs)
        with self.assertRaises(ValueError):interval_accounting([(2,1)],0,3)
        with self.assertRaises(ValueError):interval_accounting([],1,1)
        with self.assertRaises(ValueError):benchmark(lambda:0,repeats=0)


if __name__=='__main__':unittest.main()
