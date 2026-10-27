import unittest
from scheduling_lab import replay,transfer_seconds
class SchedulingTests(unittest.TestCase):
    def test_exact_single_timeline(self):
        r=replay([{'id':'x','arrival_s':1.,'prompt_tokens':2,'output_tokens':2}],2,1.,2.,0.,0.)
        self.assertEqual(r['requests'][0]['ttft_s'],4.)
        self.assertEqual(r['requests'][0]['completion_s'],6.)
        self.assertEqual(r['requests'][0]['tpot_s'],2.)
    def test_chunking_interference_and_conservation(self):
        w=[{'id':'a','arrival_s':0.,'prompt_tokens':1,'output_tokens':5},{'id':'b','arrival_s':0.004,'prompt_tokens':200,'output_tokens':2}]
        chunk,full=replay(w,8),replay(w,1000)
        a=lambda r:next(x for x in r['requests'] if x['id']=='a')
        self.assertLess(a(chunk)['tpot_s'],a(full)['tpot_s'])
        for r in (chunk,full):self.assertEqual(sum(x['emitted_tokens'] for x in r['requests']),7)
    def test_transfer_and_invalid(self):
        self.assertAlmostEqual(transfer_seconds(1e9,50e9,0.001),0.021)
        with self.assertRaises(ValueError):transfer_seconds(1,0)
        with self.assertRaises(ValueError):replay([])
