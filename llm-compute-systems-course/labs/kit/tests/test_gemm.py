import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from gemm_lab import tiled_gemm,grouped_gemm
except ImportError:torch=None


@unittest.skipIf(torch is None,'L4 uses the pinned PyTorch environment')
class GemmTests(unittest.TestCase):
    def test_irregular_dimensions_and_tile_tails(self):
        torch.manual_seed(4)
        for m,n,k,tile in [(1,3,5,2),(7,11,13,8),(2,2,1,32)]:
            a=torch.randn(m,k);b=torch.randn(k,n)
            torch.testing.assert_close(tiled_gemm(a,b,tile),a@b,rtol=1e-4,atol=1e-5)

    def test_noncontiguous_input(self):
        a=torch.randn(5,3).T;b=torch.randn(5,7)
        torch.testing.assert_close(tiled_gemm(a,b,3),a@b,rtol=1e-4,atol=1e-5)

    def test_grouped_ragged_and_empty_expert(self):
        inputs=[torch.randn(m,13) for m in (0,1,7)]
        weights=[torch.randn(13,5) for _ in inputs]
        for a,b in zip(grouped_gemm(inputs,weights,4),grouped_gemm(inputs,weights)):
            torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
        with self.assertRaises(ValueError):grouped_gemm(inputs,[])

    def test_reject_contract_mismatch(self):
        with self.assertRaises(ValueError):tiled_gemm(torch.ones(2,3),torch.ones(4,2))
        with self.assertRaises(ValueError):tiled_gemm(torch.ones(2,3),torch.ones(3,2),0)
        with self.assertRaises(ValueError):tiled_gemm(torch.ones(2,3),torch.ones(3,2,dtype=torch.float64))


if __name__=='__main__':unittest.main()
