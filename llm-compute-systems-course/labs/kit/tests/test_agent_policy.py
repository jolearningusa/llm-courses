import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import random
import unittest
from agent_policy_lab import marginal,sample,expected_tokens,wait_policy
class AgentPolicyTests(unittest.TestCase):
    def test_exact_marginal_and_zero_support(self):
        for p,q in [([.1,.3,.6],[.5,.4,.1]),([0.,1.],[1.,0.]),([.5,.5],[.5,.5])]:
            for a,b in zip(marginal(p,q),p):self.assertAlmostEqual(a,b)
    def test_sampling_corrects_rejected_mass(self):
        p,q=[.1,.3,.6],[.5,.4,.1];rng=random.Random(42);counts=[0,0,0]
        for _ in range(20000):counts[sample(p,q,rng)[0]]+=1
        for n,target in zip(counts,p):self.assertLess(abs(n/20000-target),.015)
        self.assertAlmostEqual(expected_tokens(.8,4),3.3616)
    def test_wait_policy_units_and_invalidation(self):
        self.assertEqual(wait_policy('retain',100,2,.3)['idle_hbm_byte_seconds'],200)
        self.assertEqual(wait_policy('ttl',100,2,.3,ttl_s=.1)['idle_hbm_byte_seconds'],10)
        self.assertEqual(wait_policy('ttl',100,.1,.3,ttl_s=.1)['extra_resume_seconds'],0)
        self.assertEqual(wait_policy('retain',100,2,.3,identity_valid=False)['extra_resume_seconds'],.3)
        r=wait_policy('offload',100,.01,.3,.02,.03)
        self.assertAlmostEqual(r['extra_resume_seconds'],.04)
        self.assertEqual(r['transferred_bytes'],200)
    def test_invalid(self):
        with self.assertRaises(ValueError):marginal([1.,1.],[.5,.5])
        with self.assertRaises(ValueError):wait_policy('retain',1,-1,1)
