import json
import math
import unittest
import torch
from policy import Policy
from task import EOS
from steering_ablation import (SteeringPolicy,steering,pair_indices,fit_candidates,
                               smooth_bank,corrected_loss,routes,PROTOCOL)
C=json.loads(PROTOCOL.read_text())

class TestSteering(unittest.TestCase):
    def test_zero_intervention_matches_original_policy(self):
        torch.manual_seed(1);base=Policy();candidate=SteeringPolicy();candidate.load_state_dict(base.state_dict())
        a=torch.tensor([1,3]);b=torch.tensor([2,4]);tokens=torch.tensor([[3,EOS,EOS],[7,EOS,EOS]])
        reference=base.logprobs(a,b,tokens)
        with steering(candidate,torch.zeros(2,64)):actual=candidate.logprobs(a,b,tokens)
        for x,y in zip(reference,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_intervention_context_restores_after_failure(self):
        policy=SteeringPolicy()
        with self.assertRaises(ValueError):
            with steering(policy,torch.ones(1,64)):
                raise ValueError('controlled interruption')
        self.assertIsNone(policy.direction)

    def test_pairing_is_within_prompt_and_rejects_homogeneous_group(self):
        labels=[1,0,0,1,1,0]+[1]*6+[0]*6
        pairs=pair_indices(labels,6,4)
        self.assertEqual(len(pairs),3)
        for positive,negative in pairs:
            self.assertEqual(positive//6,negative//6)
            self.assertEqual((labels[positive],labels[negative]),(1,0))
        self.assertEqual(len({i for pair in pairs for i in pair}),6)

    def test_routes_are_balanced_and_norm_matched(self):
        bank=torch.eye(2,64);r=routes(bank,12,C)
        self.assertEqual(int((r.norm(dim=1)==0).sum()),4)
        torch.testing.assert_close(r.norm(dim=1)[r.norm(dim=1)>0],torch.full((8,),C['alpha']))
        self.assertIsNone(routes(None,12,C))

    def test_pca_clusters_return_original_hidden_space_means(self):
        positive=torch.tensor([[1.,0,0],[2.,0,0],[0.,1,0],[0.,2,0]])
        config=dict(C,pca_dim=2)
        fitted=fit_candidates([(v,torch.zeros(3)) for v in positive],config)
        self.assertEqual(tuple(fitted.shape),(2,3))
        self.assertEqual({tuple(v.tolist()) for v in fitted},{(1.5,0.,0.),(0.,1.5,0.)})
        self.assertIsNone(fit_candidates([(torch.zeros(3),torch.zeros(3))]*4,config))

    def test_matching_precedes_ema(self):
        old=torch.eye(2);swapped=torch.tensor([[0.,3.],[2.,0.]])
        torch.testing.assert_close(smooth_bank(old,swapped,.5),torch.diag(torch.tensor([1.5,2.])))

    def test_actual_behavior_correction_and_old_anchor_are_distinct(self):
        old=torch.tensor([[math.log(.5),math.log(.25)]])
        behavior=torch.tensor([[math.log(.1),math.log(.5)]],requires_grad=True)
        new=old.clone().requires_grad_()
        loss,importance=corrected_loss(new,old,behavior,torch.ones(1),torch.ones(1,2),C)
        torch.testing.assert_close(importance,torch.tensor([[2.,.5]]))
        self.assertAlmostEqual(float(loss.detach()),-1.25)
        loss.backward();torch.testing.assert_close(new.grad,torch.tensor([[-1.,-.25]]))
        self.assertIsNone(behavior.grad)
        shifted=(old+.5).requires_grad_();clipped,_=corrected_loss(shifted,old,old,torch.ones(1),torch.ones(1,2),C)
        clipped.backward();torch.testing.assert_close(shifted.grad,torch.zeros_like(shifted))

    def test_split_has_no_reversed_operand_leak(self):
        groups=lambda ps:{tuple(sorted(p)) for p in ps}
        self.assertFalse(groups(C['train_prompts']) & groups(C['heldout_prompts']))
        self.assertTrue(set(map(tuple,C['sft_prompts']))<=set(map(tuple,C['train_prompts'])))
        self.assertEqual(len(C['conditions'])*len(C['seeds']),9)

if __name__=='__main__':unittest.main()
