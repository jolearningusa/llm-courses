import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reductions import rmsnorm,softmax


class ReductionTests(unittest.TestCase):
    def test_rmsnorm_zero_and_tail_lengths(self):
        for length in (1,33,257):
            self.assertEqual(rmsnorm([0]*length,[1]*length),[0]*length)
            row=[math.sin(i*.13) for i in range(length)]
            weights=[1+i/length for i in range(length)]
            out=rmsnorm(row,weights)
            denominator=math.sqrt(sum(x*x for x in row)/length+1e-5)
            for y,x,w in zip(out,row,weights):self.assertAlmostEqual(y,x*w/denominator,places=12)

    def test_stable_softmax_and_shift_invariance(self):
        a=softmax([1000,1001]);b=softmax([-1,0])
        self.assertAlmostEqual(sum(a),1)
        for x,y in zip(a,b):self.assertAlmostEqual(x,y)
        self.assertAlmostEqual(a[0],1/(1+math.e))

    def test_invalid_contracts(self):
        for row in ([],[float('nan')],[float('inf')],[True]):
            with self.assertRaises(ValueError):softmax(row)
        with self.assertRaises(ValueError):rmsnorm([1,2],[1])
        with self.assertRaises(ValueError):rmsnorm([1],[1],eps=0)


if __name__=='__main__':unittest.main()
