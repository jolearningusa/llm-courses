import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    import torch
    from attention_lab import full_attention,tiled_attention
except ImportError:torch=None


@unittest.skipIf(torch is None,'L5 uses the pinned PyTorch environment')
class AttentionTests(unittest.TestCase):
    def test_masks_irregular_tiles_and_single_token(self):
        torch.manual_seed(8)
        for length in (1,7,17):
            q=torch.randn(length,3,dtype=torch.float64);k=torch.randn_like(q);v=torch.randn(length,5,dtype=q.dtype)
            for causal in (False,True):
                torch.testing.assert_close(tiled_attention(q,k,v,causal,3,4),full_attention(q,k,v,causal))

    def test_backward_matches_full(self):
        torch.manual_seed(2)
        inputs=[torch.randn(7,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        expected=full_attention(*inputs,causal=True).square().sum()
        reference=torch.autograd.grad(expected,inputs)
        actual=tiled_attention(*inputs,causal=True,tile_q=3,tile_k=2).square().sum()
        gradients=torch.autograd.grad(actual,inputs)
        for a,b in zip(gradients,reference):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-8)

    def test_causal_output_ignores_future_values(self):
        q=torch.ones(5,2);k=torch.ones_like(q);v=torch.randn_like(q)
        a=tiled_attention(q,k,v,True);v[4]+=100
        b=tiled_attention(q,k,v,True)
        torch.testing.assert_close(a[:4],b[:4])

    def test_contract_rejected(self):
        q=torch.ones(3,2)
        with self.assertRaises(ValueError):tiled_attention(q,q,q,tile_k=0)
        with self.assertRaises(ValueError):full_attention(q,q[:2],q)


if __name__=='__main__':unittest.main()
