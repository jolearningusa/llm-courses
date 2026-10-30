"""L3T row-wise fused softmax and RMSNorm in Triton, checked against PyTorch.

On a CUDA/ROCm GPU it times fused kernels against eager and library paths. With
--interpret it runs the same kernels in Triton's CPU interpreter for correctness
only; interpreter timings say nothing about GPU performance and are not reported.
"""
import argparse
import json
import os
import sys

DTYPES={'float32','float16','bfloat16'}
MAX_COLS=1<<16  # one program holds the whole row in registers; longer rows need a multi-pass kernel


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interpret',action='store_true',help='run kernels in the Triton CPU interpreter (correctness only)')
    parser.add_argument('--rows',type=int,default=None,help='rows per shape (default 4096 on GPU, 64 interpreted)')
    parser.add_argument('--cols',type=int,nargs='+',default=None,help='row lengths to sweep')
    parser.add_argument('--dtype',choices=sorted(DTYPES),default='float16')
    parser.add_argument('--num-warps',type=int,default=None,help='override the heuristic warp count')
    args=parser.parse_args(argv)
    if args.rows is not None and args.rows<1:parser.error('rows must be positive')
    if args.cols and any(c<1 or c>MAX_COLS for c in args.cols):parser.error(f'cols must be in 1..{MAX_COLS}')
    if args.num_warps is not None and args.num_warps not in (1,2,4,8,16,32):parser.error('num-warps must be a power of two up to 32')
    return args


# The interpreter is chosen when kernels are decorated, so the flag must be read before importing triton.
if __name__=='__main__':
    ARGS=parse_args()
    if ARGS.interpret:os.environ['TRITON_INTERPRET']='1'

import torch
import triton
import triton.language as tl


def interpreting():
    return os.environ.get('TRITON_INTERPRET')=='1'


@triton.jit
def softmax_kernel(out_ptr,in_ptr,in_row_stride,out_row_stride,n_cols,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.arange(0,BLOCK);mask=cols<n_cols
    x=tl.load(in_ptr+row*in_row_stride+cols,mask=mask,other=-float('inf')).to(tl.float32)
    x=x-tl.max(x,axis=0)  # padded lanes stay -inf and contribute exp(-inf)=0
    numerator=tl.exp(x)
    y=numerator/tl.sum(numerator,axis=0)
    tl.store(out_ptr+row*out_row_stride+cols,y.to(out_ptr.dtype.element_ty),mask=mask)


@triton.jit
def rmsnorm_kernel(out_ptr,in_ptr,weight_ptr,in_row_stride,out_row_stride,n_cols,eps,BLOCK:tl.constexpr):
    row=tl.program_id(0).to(tl.int64)
    cols=tl.arange(0,BLOCK);mask=cols<n_cols
    x=tl.load(in_ptr+row*in_row_stride+cols,mask=mask,other=0.0).to(tl.float32)
    w=tl.load(weight_ptr+cols,mask=mask,other=0.0).to(tl.float32)
    inv=1.0/tl.sqrt(tl.sum(x*x,axis=0)/n_cols+eps)
    tl.store(out_ptr+row*out_row_stride+cols,(x*inv*w).to(out_ptr.dtype.element_ty),mask=mask)


def check_rows(x):
    if not isinstance(x,torch.Tensor) or x.ndim!=2:raise ValueError('a 2-D (rows, cols) tensor is required')
    if min(x.shape)<1:raise ValueError('nonempty rows and columns required')
    if x.shape[1]>MAX_COLS:raise ValueError(f'rows longer than {MAX_COLS} need a multi-pass kernel')
    if str(x.dtype).removeprefix('torch.') not in DTYPES:raise ValueError('float32, float16 or bfloat16 required')
    if x.stride(1)!=1:raise ValueError('the last dimension must be contiguous; row stride may be anything')
    if interpreting()!=(x.device.type=='cpu'):raise ValueError('use CPU tensors with TRITON_INTERPRET=1 and GPU tensors otherwise')


def launch_config(n_cols,num_warps=None):
    block=triton.next_power_of_2(n_cols)
    if num_warps is None:num_warps=4 if block<=2048 else 8 if block<=8192 else 16
    return block,num_warps


def softmax(x,num_warps=None):
    check_rows(x)
    out=torch.empty(x.shape,dtype=x.dtype,device=x.device)
    block,warps=launch_config(x.shape[1],num_warps)
    softmax_kernel[(x.shape[0],)](out,x,x.stride(0),out.stride(0),x.shape[1],BLOCK=block,num_warps=warps)
    return out


def rmsnorm(x,weight,eps=1e-6,num_warps=None):
    check_rows(x)
    if weight.ndim!=1 or weight.shape[0]!=x.shape[1] or weight.device!=x.device or weight.stride(0)!=1:
        raise ValueError('weight must be a contiguous vector of row length on the same device')
    if not eps>0:raise ValueError('eps must be positive')
    out=torch.empty(x.shape,dtype=x.dtype,device=x.device)
    block,warps=launch_config(x.shape[1],num_warps)
    rmsnorm_kernel[(x.shape[0],)](out,x,weight,x.stride(0),out.stride(0),x.shape[1],eps,BLOCK=block,num_warps=warps)
    return out


def reference_softmax(x):
    return torch.softmax(x.float(),dim=-1).to(x.dtype)


def reference_rmsnorm(x,weight,eps=1e-6):
    x32=x.float()
    return (x32*torch.rsqrt(x32.square().mean(dim=-1,keepdim=True)+eps)*weight.float()).to(x.dtype)


def unfused_softmax(x):
    """Eager softmax written as separate ops: each line is a kernel and a round trip to memory."""
    shifted=x-x.max(dim=-1,keepdim=True).values
    numerator=shifted.exp()
    return numerator/numerator.sum(dim=-1,keepdim=True)


def tolerance(dtype):
    return {torch.float32:(1e-5,1e-6),torch.float16:(1e-3,1e-3),torch.bfloat16:(1.6e-2,1e-2)}[dtype]


def ideal_bytes(rows,cols,dtype):
    """One read of the input and one write of the output: the fused-kernel floor."""
    return 2*rows*cols*torch.empty((),dtype=dtype).element_size()


def time_ms(fn):
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode='median')  # event-timed repeats, cache cleared before each


def run(rows,cols_list,dtype_name,num_warps=None):
    dtype=getattr(torch,dtype_name)
    device='cpu' if interpreting() else 'cuda'
    torch.manual_seed(3)
    rtol,atol=tolerance(dtype);results=[]
    for cols in cols_list:
        base=(torch.randn(rows,cols+3,device=device,dtype=torch.float32)*4).to(dtype)
        x=base[:,:cols]  # row stride differs from cols, which checks the stride arguments
        weight=torch.rand(cols,device=device,dtype=dtype)+0.5
        torch.testing.assert_close(softmax(x,num_warps),reference_softmax(x),rtol=rtol,atol=atol)
        torch.testing.assert_close(rmsnorm(x,weight,num_warps=num_warps),reference_rmsnorm(x,weight),rtol=rtol,atol=atol)
        entry={'cols':cols,'block':launch_config(cols,num_warps)[0],'num_warps':launch_config(cols,num_warps)[1],'checks':'passed'}
        if not interpreting():
            xc=x.contiguous();gb=ideal_bytes(rows,cols,dtype)/1e9
            timings={'triton_softmax':time_ms(lambda:softmax(xc,num_warps)),
                     'torch_softmax':time_ms(lambda:torch.softmax(xc,dim=-1)),
                     'eager_unfused_softmax':time_ms(lambda:unfused_softmax(xc)),
                     'triton_rmsnorm':time_ms(lambda:rmsnorm(xc,weight,num_warps=num_warps)),
                     'torch_rms_norm':time_ms(lambda:torch.nn.functional.rms_norm(xc,(cols,),weight,1e-6))}
            entry['median_ms']=timings
            entry['effective_GBps_at_ideal_bytes']={k:gb/(v/1e3) for k,v in timings.items()}
        results.append(entry)
    environment={'torch':torch.__version__,'triton':triton.__version__,'python':sys.version.split()[0]}
    if interpreting():
        provenance='Triton CPU interpreter: numerical checks only, no timing'
    else:
        environment['device']=torch.cuda.get_device_name()
        environment['runtime']='hip '+torch.version.hip if torch.version.hip else 'cuda '+str(torch.version.cuda)
        provenance='GPU: triton.testing.do_bench median ms; effective bandwidth assumes one read and one write per element'
    return {'schema_version':1,'lab':'L3T','provenance':provenance,'environment':environment,
            'rows':rows,'dtype':dtype_name,'results':results,
            'limits':'forward only; whole row per program (cols <= %d); bandwidth uses ideal bytes, not profiler counters'%MAX_COLS}


if __name__=='__main__':
    if not ARGS.interpret and not torch.cuda.is_available():
        print('No CUDA/ROCm device: rerun with --interpret for the CPU correctness path.',file=sys.stderr);sys.exit(77)
    rows=ARGS.rows or (64 if ARGS.interpret else 4096)
    cols=ARGS.cols or ([1,33,257,1000] if ARGS.interpret else [128,1000,4096,16384])
    print(json.dumps(run(rows,cols,ARGS.dtype,ARGS.num_warps),indent=2,allow_nan=False))
