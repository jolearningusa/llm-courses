"""Reuse course checks to separate derivatives, shapes and decoder causality.

Run from a full llm-courses checkout with NumPy; no model training or GPU. The
mask mutation is temporary and in-process: original source files are unchanged.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest
from unittest.mock import patch

import numpy as np

from autograd import Tensor
from test_mlp import Autograd, numerical_grad, rel_err

ROOT = Path(__file__).resolve().parents[3]
ARCH = ROOT / 'llm-transformer-architecture-course/labs/L1_one_block'
sys.path.insert(0, str(ARCH))
import transformer
from test_transformer import OneBlockThreeFamilies


def run_cases(cases):
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(cases))
    return dict(tests=result.testsRun, passed=result.testsRun-len(result.failures)-len(result.errors),
                failures=[dict(test=case.id(), trace=trace.replace(str(ROOT), '<repo>'))
                          for case,trace in result.failures],
                errors=[dict(test=case.id(), trace=trace.replace(str(ROOT), '<repo>'))
                        for case,trace in result.errors],
                successful=result.wasSuccessful())


def residual_check():
    rng = np.random.default_rng(7)
    x = Tensor(rng.normal(size=(2,3)))
    w = Tensor(rng.normal(scale=.2, size=(3,3)))
    y = x + (x @ w).tanh()
    y.sum().backward()
    errors = {}
    for name, p in [('input',x),('weight',w)]:
        ref = numerical_grad(lambda: float((x + (x @ w).tanh()).sum().data), p.data)
        errors[name] = rel_err(ref, p.grad)
        if errors[name] >= 1e-6:
            raise AssertionError('smooth residual derivative disagrees with centered differences')
    # Identity path in the zero-branch limit, independent of finite-difference error.
    z = Tensor(np.array([[1.,2.,3.]]))
    (z + z @ Tensor(np.zeros((3,3)))).sum().backward()
    np.testing.assert_array_equal(z.grad, np.ones((1,3)))
    return dict(input_shape=list(x.shape), branch_shape=list(y.shape),
                formula='y=x+tanh(x@W), loss=sum(y)', seed=7, step=1e-6,
                max_relative_error=errors, identity_path_gradient=z.grad.tolist())


def prefix_delta():
    model = transformer.TinyTransformer(seed=0)
    tokens = [3,14,15,9,26,5,35,8]
    changed = tokens[:5]+[1,2,4]
    a,b = model.decode(tokens),model.decode(changed)
    return dict(shape=list(a.shape), changed_positions=[5,6,7],
                prefix_max_abs_delta=float(np.max(np.abs(a[:5]-b[:5]))))


def report():
    loader = unittest.TestLoader()
    baseline = run_cases(list(loader.loadTestsFromTestCase(Autograd)) +
                         list(loader.loadTestsFromTestCase(OneBlockThreeFamilies)))
    residual = residual_check()
    causal = prefix_delta()
    # Reuse the exact existing oracle. Only mask construction is changed, and
    # patch's context restores it even if a check raises.
    with patch.object(transformer, 'causal_mask', lambda n: np.ones((n,n), dtype=bool)):
        mutant = run_cases([OneBlockThreeFamilies('test_attention_matches_definition'),
                            OneBlockThreeFamilies('test_decoder_never_sees_the_future')])
        leaky = prefix_delta()
    restored = run_cases([OneBlockThreeFamilies('test_decoder_never_sees_the_future')])
    wanted = 'test_transformer.OneBlockThreeFamilies.test_decoder_never_sees_the_future'
    rejected = (mutant['tests']==2 and mutant['passed']==1 and not mutant['errors']
                and [f['test'] for f in mutant['failures']]==[wanted])
    if not (baseline['successful'] and rejected and restored['successful']
            and causal['prefix_max_abs_delta']==0 and leaky['prefix_max_abs_delta']>1e-6):
        raise AssertionError('bridge controls did not produce their declared outcomes')
    paths = [Path(__file__), Path(__file__).with_name('autograd.py'),
             Path(__file__).with_name('mlp.py'), Path(__file__).with_name('test_mlp.py'),
             ARCH/'transformer.py', ARCH/'test_transformer.py']
    return dict(schema=1, python=platform.python_version(), numpy=np.__version__,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths},
                baseline=baseline, residual=residual, causal=causal, unmasked=leaky,
                mutant=mutant, expected_mutant_rejected=rejected, restored=restored,
                limits='Random-weight CPU fixtures, no learned quality, new training run, '
                       'GPU timing, universal gradient tolerance or full-course qualification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(report(),stream,indent=2,sort_keys=True)
        stream.write('\n')


if __name__=='__main__':main()
