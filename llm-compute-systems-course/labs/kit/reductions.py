"""L3 serial numerical references; not accelerated performance baselines."""
import argparse
import json
import math


def finite_row(values):
    row=list(values)
    if not row or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in row):
        raise ValueError('row must contain finite numbers')
    return row


def rmsnorm(values,weights,eps=1e-5):
    row=finite_row(values);weights=finite_row(weights)
    if len(row)!=len(weights):raise ValueError('weight and row lengths differ')
    if not isinstance(eps,(int,float)) or isinstance(eps,bool) or not math.isfinite(eps) or eps<=0:raise ValueError('eps must be finite and positive')
    inv=1/math.sqrt(math.fsum(x*x for x in row)/len(row)+eps)
    return [x*w*inv for x,w in zip(row,weights)]


def softmax(values):
    row=finite_row(values);maximum=max(row)
    exps=[math.exp(x-maximum) for x in row];total=math.fsum(exps)
    return [x/total for x in exps]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cols',type=int,default=257)
    args=parser.parse_args()
    if args.cols<1:parser.error('cols must be positive')
    row=[math.sin(i*.13) for i in range(args.cols)]
    print(json.dumps({'lab':'L3','provenance':'CPU serial reference; no timing',
        'cols':args.cols,'rmsnorm_first_values':rmsnorm(row,[1]*args.cols)[:5],
        'stable_softmax_large_logits':softmax([1000,1001])},indent=2))
