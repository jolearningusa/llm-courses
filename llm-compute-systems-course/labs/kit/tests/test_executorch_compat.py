import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).resolve().parents[1]/'executorch_compat.py'
spec=importlib.util.spec_from_file_location('executorch_compat',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TestExecuTorchCompatibility(unittest.TestCase):
    def test_scalar_oracle_relu_and_bias(self):
        fixture={'weight1':[[1.,-1.],[0.,1.]],'bias1':[0.,1.],
                 'weight2':[[2.,3.]],'bias2':[-1.]}
        # [2,1] -> hidden [1,2] -> 2 + 6 - 1 = 7;
        # [0,-2] -> hidden [2,0] -> 4 - 1 = 3.
        self.assertEqual(module.oracle(fixture,[[2.,1.],[0.,-2.]]),[[7.],[3.]])

    def test_bad_inputs_do_not_call_runtime(self):
        record=module.input_guards()
        self.assertEqual(set(record['rejected']),{'shape','dtype','stride'})
        self.assertEqual(record['native_calls_for_invalid_inputs'],0)

if __name__=='__main__':unittest.main()
