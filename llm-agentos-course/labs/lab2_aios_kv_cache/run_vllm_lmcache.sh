#!/usr/bin/env bash
# Lab 2: serve a model with LMCache as vLLM's KV offloading backend and a 64 GiB CPU DRAM tier.
#
# Reference (checked Oct 2026): https://docs.vllm.ai/en/stable/examples/disaggregated/lmcache/
#   --kv-offloading-backend lmcache   use LMCache for KV offload
#   --kv-offloading-size <GiB>        size of the CPU offload tier
#
# Requirements: pip install vllm lmcache; one CUDA GPU; at least ~80 GiB free host RAM for a 64 GiB tier.
# Usage: ./run_vllm_lmcache.sh [model] [cpu_gib]
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-8B}"
CPU_GIB="${2:-64}"
PORT="${PORT:-8000}"

# 256-token chunks let LMCache reuse KV for a repeated span anywhere in the prompt, not only a shared prefix.
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"

echo "Serving ${MODEL} on :${PORT} with ${CPU_GIB} GiB LMCache CPU offload (chunk=${LMCACHE_CHUNK_SIZE})"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --kv-offloading-backend lmcache \
  --kv-offloading-size "${CPU_GIB}" \
  --disable-hybrid-kv-cache-manager \
  --enable-prefix-caching

# Try it from another shell: send the same long system prompt twice and compare latency.
#   curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
#     -d '{"model":"Qwen/Qwen3-8B","prompt":"<long context> Question: ...","max_tokens":1}'
