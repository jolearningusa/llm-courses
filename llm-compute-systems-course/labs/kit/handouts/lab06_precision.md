# L6 — Precision contracts and measured numerical behavior

**Released:** packed int4/int8 CPU reference and CPU BF16 autocast experiment, Python 3.12/PyTorch 2.9.1. **Unverified:** TorchAO/Transformer Engine accelerated kernels and LLM quality. Estimated duration 4–6 hours, student pilot pending.

From `labs/kit/` run:

```sh
python3 precision_lab.py > precision-packing.json
python3 precision_lab.py --autocast > precision-autocast.json
python3 -m unittest discover -s tests -v
```

The packing path uses only the standard library; autocast needs the pinned PyTorch environment. The supplied complete reference is an inspection baseline. Student task: first derive and implement a decoder independently, then compare with `dequantize`.

## A. Bytes and reconstruction

Derive 640 B for 1,024 int4 weights with group size 32 and FP32 scales. Compare group sizes 4/32/128 on smooth and outlier-heavy values. Include tail padding and scale bytes; exclude Python object overhead only if stated. The format uses signed two's-complement nibbles, low nibble first, qmax=7 or 127, FP32 little-endian scales and zero-tail padding. It is not interchangeable with TorchAO, GGUF or vendor layouts. Test zeros, negative values, uneven groups and rejected nonfinite/unrepresentable scales. Verify scalar error ≤scale/2 plus FP32 rounding tolerance; this is not a task-quality guarantee.

## B. One supported recipe

Part B is the training-precision exercise. It uses the same environment but is done with deck 25 (training precision, DDP and FSDP) in week 13; Parts A and C are due in week 7.

Inspect `autocast_comparison`: identical initial model, data, optimizer and five steps under FP32 or CPU BF16 autocast. Report each loss, output dtype and final gradient difference. Require finite losses/gradients and compare final loss within a predeclared 2% relative threshold for this fixture; investigate rejection rather than silently loosening it. This tiny synthetic regression validates numerical plumbing, not LLM convergence or speed. No timing claim is made. Change input magnitude or shape and explain behavior.

## C. Target-specific extension

Select one compatible TorchAO or Transformer Engine release and target. Record torch/library/device/driver versions, packing, group size, operand/accumulator/output dtypes, fallback and actual kernel. Keep calibration and held-out evaluation separate for PTQ, or match training tokens/optimizer/schedule for QAT/low-precision training. Measure conversion, packing, cold start and warm execution separately, including all costs in end-to-end accounting. Include a tiny or unsupported shape and an outlier workload. Do not install latest TorchAO into the pinned baseline without checking compatibility.

Submit raw JSON, a byte/error table, the declared numerical threshold and the [report](../../report_template.md). No vendor/GPU experiment ran on the authoring host. Instructor diagnostic: storing codes as Python integers does not demonstrate packed storage; dequantizing into FP32 before matmul demonstrates reconstruction only.
