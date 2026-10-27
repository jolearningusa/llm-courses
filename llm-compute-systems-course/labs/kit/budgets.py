"""Analytical resource budgets, not hardware benchmarks. Python standard library only."""
import argparse
import json
import math


def positive(name, value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be numeric')
    if not math.isfinite(value) or value <= 0 or (integer and int(value) != value):
        raise ValueError(name + ' must be a positive ' + ('integer' if integer else 'finite number'))
    return value


def gemm(m, n, k, element_bytes=2, compute_tflops=100, bandwidth_tbps=1):
    """Ideal input reads + output write; decimal rate units; no launch/latency model."""
    for name, value in [('m', m), ('n', n), ('k', k)]:
        positive(name, value, integer=True)
    for name, value in [('element_bytes', element_bytes), ('compute_tflops', compute_tflops), ('bandwidth_tbps', bandwidth_tbps)]:
        positive(name, value)
    flops = 2 * m * n * k
    traffic = element_bytes * (m*k + k*n + m*n)
    math_s = flops / (compute_tflops * 1e12)
    memory_s = traffic / (bandwidth_tbps * 1e12)
    return {'flops': flops, 'ideal_traffic_bytes': traffic,
            'arithmetic_intensity': flops / traffic,
            'math_bound_seconds': math_s, 'memory_bound_seconds': memory_s,
            'lower_bound_seconds': max(math_s, memory_s),
            'limiting_term': 'compute' if math_s >= memory_s else 'memory',
            'provenance': 'analytical; rates are supplied assumptions; not measured'}


def kv_bytes(layers, kv_heads, head_dim, lengths, element_bytes=2):
    """Logical MHA/GQA K+V bytes; no paging, metadata, replicas or workspace."""
    for name, value in [('layers', layers), ('kv_heads', kv_heads), ('head_dim', head_dim)]:
        positive(name, value, integer=True)
    positive('element_bytes', element_bytes)
    lengths = list(lengths)
    for length in lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError('sequence lengths must be nonnegative integers')
    return 2 * layers * kv_heads * head_dim * element_bytes * sum(lengths)


def training_state(parameters, ranks=1, stage=0, parameter_bytes=2,
                   gradient_bytes=2, optimizer_bytes=12):
    """Persistent per-rank state only. Optimizer bytes include any master copy."""
    positive('parameters', parameters, integer=True)
    positive('ranks', ranks, integer=True)
    if isinstance(stage, bool) or stage not in (0, 1, 2, 3):
        raise ValueError('stage must be 0, 1, 2, or 3')
    for name, value in [('parameter_bytes', parameter_bytes), ('gradient_bytes', gradient_bytes), ('optimizer_bytes', optimizer_bytes)]:
        positive(name, value)
    p = parameters * parameter_bytes / (ranks if stage == 3 else 1)
    g = parameters * gradient_bytes / (ranks if stage >= 2 else 1)
    o = parameters * optimizer_bytes / (ranks if stage >= 1 else 1)
    return {'parameter_bytes_per_rank': p, 'gradient_bytes_per_rank': g,
            'optimizer_bytes_per_rank': o, 'persistent_bytes_per_rank': p+g+o,
            'excluded': 'activations, transient gathers, workspace, padding and allocator reserve'}


def demo():
    return {'schema_version': 1, 'lab': 'L0', 'provenance': 'analytical',
            'large_gemm': gemm(4096,4096,4096), 'decode_gemm': gemm(1,4096,4096),
            'kv_example_bytes': kv_bytes(32,8,128,[32768]),
            'training_1b_8ranks': {str(stage): training_state(1_000_000_000,8,stage) for stage in range(4)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='Print the worked analytical fixtures')
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--compute-tflops', type=float, default=100)
    parser.add_argument('--bandwidth-tbps', type=float, default=1)
    args = parser.parse_args()
    try:
        result = demo() if args.demo else gemm(args.m,args.n,args.k,compute_tflops=args.compute_tflops,bandwidth_tbps=args.bandwidth_tbps)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
