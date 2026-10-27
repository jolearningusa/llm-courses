import math
import struct
import unittest
from precision_lab import quantize, dequantize

class PrecisionTests(unittest.TestCase):
    def test_zero_and_odd_tail(self):
        for bits in (4,8):
            p=quantize([0.,0.,0.],bits,2)
            self.assertEqual(dequantize(p),[0.,0.,0.])
            self.assertEqual(len(p.payload),2 if bits==4 else 3)
            self.assertEqual(len(p.scales),8)
    def test_signed_packing(self):
        p=quantize([-7.,7.,0.],4,3)
        self.assertEqual(p.payload,bytes([0x79,0x00]))
        self.assertEqual(dequantize(p),[-7.,7.,0.])
    def test_error_with_outlier_and_tails(self):
        values=[math.sin(i) for i in range(65)]+[20.]
        for bits in (4,8):
            p=quantize(values,bits,7)
            scales=struct.unpack('<'+'f'*(len(p.scales)//4),p.scales)
            for i,(x,y) in enumerate(zip(values,dequantize(p))):
                self.assertLessEqual(abs(x-y),scales[i//7]/2+1e-5)
    def test_invalid(self):
        for values,bits,g in [([],4,2),([float('nan')],4,2),([1],3,2),([1],4,0),([1e100],4,2)]:
            with self.assertRaises(ValueError):quantize(values,bits,g)
