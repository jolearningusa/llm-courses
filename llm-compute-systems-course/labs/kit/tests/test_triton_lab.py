import os
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
    if not torch.cuda.is_available():os.environ['TRITON_INTERPRET']='1'  # must precede the kernel definitions
    import triton  # noqa: F401
    import triton_lab as lab
except ImportError:lab=None


@unittest.skipIf(lab is None,'L3T needs the Triton environment (requirements-triton.txt)')
class TritonLabTests(unittest.TestCase):
    def setUp(self):
        self.device='cpu' if lab.interpreting() else 'cuda'
        torch.manual_seed(11)

    def rows(self,rows,cols,dtype=None,scale=1.0):
        return torch.randn(rows,cols,device=self.device).mul(scale).to(dtype or torch.float32)

    def test_softmax_tails_and_strided_rows(self):
        for cols in (1,2,33,257,1000):
            base=self.rows(5,cols+7)
            x=base[:,3:3+cols]
            torch.testing.assert_close(lab.softmax(x),lab.reference_softmax(x))

    def test_softmax_large_logits_are_stable(self):
        x=torch.tensor([[1000.0,1001.0],[-1000.0,-1001.0]],device=self.device)
        expected=torch.tensor([[0.268941,0.731059],[0.731059,0.268941]],device=self.device)
        torch.testing.assert_close(lab.softmax(x),expected,rtol=0,atol=1e-6)

    def test_softmax_rows_sum_to_one_in_low_precision(self):
        for dtype in (torch.float16,torch.bfloat16):
            x=self.rows(4,300,dtype,scale=6)
            rtol,atol=lab.tolerance(dtype)
            out=lab.softmax(x)
            torch.testing.assert_close(out,lab.reference_softmax(x),rtol=rtol,atol=atol)
            torch.testing.assert_close(out.float().sum(-1),torch.ones(4,device=self.device),rtol=0,atol=1e-2)

    def test_rmsnorm_matches_reference_with_nonuniform_weight(self):
        for cols in (1,33,257):
            x=self.rows(6,cols);w=torch.rand(cols,device=self.device)+0.5
            torch.testing.assert_close(lab.rmsnorm(x,w),lab.reference_rmsnorm(x,w))

    def test_rmsnorm_zero_row_is_zero(self):
        x=torch.zeros(2,17,device=self.device);w=torch.ones(17,device=self.device)
        self.assertEqual(lab.rmsnorm(x,w).abs().max().item(),0.0)

    def test_contract_is_rejected(self):
        x=self.rows(4,8)
        with self.assertRaises(ValueError):lab.softmax(x.t())  # last dimension not contiguous
        with self.assertRaises(ValueError):lab.softmax(x[0])
        with self.assertRaises(ValueError):lab.softmax(x.double())
        with self.assertRaises(ValueError):lab.rmsnorm(x,torch.ones(7,device=self.device))
        with self.assertRaises(ValueError):lab.rmsnorm(x,torch.ones(8,device=self.device),eps=0)

    def test_run_records_checks_without_interpreter_timings(self):
        report=lab.run(3,[5,64],'float32')
        self.assertEqual([r['checks'] for r in report['results']],['passed','passed'])
        if lab.interpreting():self.assertNotIn('median_ms',report['results'][0])


if __name__=='__main__':unittest.main()
