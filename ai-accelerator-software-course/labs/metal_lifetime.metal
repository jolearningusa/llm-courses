#include <metal_stdlib>
using namespace metal;
kernel void lifetime_probe(device const uint *input [[buffer(0)]],
                           device uint *output [[buffer(1)]],
                           device atomic_uint *coverage [[buffer(2)]],
                           constant uint &n [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    output[i] = (input[i] ^ 0xA5A5A5A5u) + i;
    atomic_fetch_add_explicit(coverage, 1u, memory_order_relaxed);
}
