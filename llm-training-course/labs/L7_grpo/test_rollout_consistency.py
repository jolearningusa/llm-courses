import math
import unittest

import torch

from policy import Policy
from task import EOS
from rollout_consistency import check_same_contract, fingerprint, replay, report, valid_ratios


class RolloutConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.policy=Policy(hidden=8).double().eval()
        self.a,self.b=torch.tensor([1,2]),torch.tensor([2,3])
        self.tokens=torch.tensor([[3,EOS,EOS,EOS],[4,5,EOS,EOS]])
        self.lp,self.mask=self.policy.logprobs(self.a,self.b,self.tokens)

    def test_stepwise_and_batched_same_policy_ratio_is_one(self):
        old=replay(self.policy,self.a,self.b,self.tokens)
        self.assertLess(check_same_contract(self.lp,old,self.mask,self.mask),1e-12)

    def test_eos_included_padding_excluded_and_mask_mismatch_rejected(self):
        self.assertEqual(self.mask.tolist(),[[1,1,0,0],[1,1,1,0]])
        wrong=self.mask.clone();wrong[0,1]=0
        with self.assertRaises(ValueError):check_same_contract(self.lp,self.lp,wrong,self.mask)

    def test_bad_padding_never_enters_exponent(self):
        previous=self.lp.detach().clone();previous[~self.mask.bool()]=float('nan')
        torch.testing.assert_close(valid_ratios(self.lp,previous,self.mask),torch.ones(5,dtype=torch.float64))

    def test_temperature_or_previous_token_mismatch_breaks_identity(self):
        for kwargs in ({'temperature':.7},{'wrong_shift':True}):
            with self.subTest(**kwargs),self.assertRaises(AssertionError):
                check_same_contract(self.lp,replay(self.policy,self.a,self.b,self.tokens,**kwargs),self.mask,self.mask)

    def test_token_and_sequence_ratios_have_different_units(self):
        delta=torch.tensor([[math.log(2),math.log(3)]],dtype=torch.float64)
        ratios=valid_ratios(delta,torch.zeros_like(delta),torch.ones_like(delta))
        torch.testing.assert_close(ratios,torch.tensor([2.,3.],dtype=torch.float64))
        self.assertAlmostEqual(delta.sum().exp().item(),6.)
        self.assertNotAlmostEqual(delta.mean().exp().item(),6.)

    def test_empty_nonbinary_or_misaligned_masks_rejected(self):
        for mask in (torch.zeros_like(self.mask),self.mask+.5,self.mask[:,:2]):
            with self.assertRaises(ValueError):valid_ratios(self.lp,self.lp,mask)

    def test_fingerprint_tracks_weight_change(self):
        before=fingerprint(self.policy)
        with torch.no_grad():next(self.policy.parameters()).add_(.01)
        self.assertNotEqual(before,fingerprint(self.policy))

    def test_actual_sampling_and_controlled_update_report(self):
        data=report()
        self.assertEqual(data['versions']['behavior'],data['versions']['anchor'])
        self.assertNotEqual(data['versions']['before'],data['versions']['after'])
        self.assertLess(data['same_anchor_rollout_max_ratio_error'],1e-12)
        self.assertEqual(data['verl_execution'],'not_run')


if __name__=='__main__':unittest.main()
