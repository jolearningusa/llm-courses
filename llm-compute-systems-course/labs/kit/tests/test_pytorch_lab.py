import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from pytorch_lab import block,execute,compare,saved_state
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'L2 requires its pinned PyTorch environment')
class PyTorchLabTests(unittest.TestCase):
    def test_forward_backward_and_no_accumulation(self):
        torch.manual_seed(3);model=block();other=copy.deepcopy(model)
        x=torch.randn(4,32,requires_grad=True)
        self.assertTrue(compare(model,other,x,True)['passed'])
        execute(model,x,True);first=[p.grad.clone() for p in model.parameters()]
        execute(model,x,True)
        for p,g in zip(model.parameters(),first):torch.testing.assert_close(p.grad,g)

    def test_changed_weight_is_detected(self):
        model=block();other=copy.deepcopy(model)
        with torch.no_grad():other[0].weight[0,0].add_(.5)
        with self.assertRaises(AssertionError):compare(model,other,torch.randn(4,32),False)

    def test_saved_records_are_explicit(self):
        r=saved_state(block(),torch.randn(4,32,requires_grad=True))
        self.assertTrue(r['records'])
        self.assertGreater(r['summed_logical_bytes'],0)


if __name__=='__main__':unittest.main()
