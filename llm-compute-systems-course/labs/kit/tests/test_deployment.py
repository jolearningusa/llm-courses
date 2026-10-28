import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
try:
    import torch
except ImportError:
    torch=None
from deployment_lab import fixture,export_check,checked_forward
@unittest.skipIf(torch is None,'pinned PyTorch environment required')
class DeploymentTests(unittest.TestCase):
    def test_export_roundtrip_and_static_guard(self):
        model,x=fixture();ep,_=export_check(model,x)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'graph.pt2');torch.export.save(ep,path)
            loaded=torch.export.load(path)
            torch.testing.assert_close(loaded.module()(x),model(x))
    def test_loader_rejects_before_execution(self):
        called=[]
        call=lambda x:called.append(x)
        for x in (torch.zeros(4,8),torch.zeros(3,8,dtype=torch.float64),torch.zeros(8,3).t()):
            with self.assertRaises(ValueError):checked_forward(call,x)
        self.assertFalse(called)
