"""
Lab 2: the system memory plane, declared as a vLLM KVTransferConfig.

MultiConnector wraps several KV connectors at once:
  - save: every child connector stores the KV cache it is given
  - load: the FIRST child (in list order) that reports matching tokens serves the request

So the order below means "reuse KV from local CPU DRAM via LMCache if we have it; otherwise
receive it from the prefill instance over NIXL".

  LMCacheConnectorV1  - chunked KV reuse (256-token chunks, so a repeated block in the middle of a
                        prompt can hit, not only a shared prefix) offloaded to a CPU DRAM tier
  NixlConnector       - GPU-to-GPU KV transfer between a prefill and a decode instance
                        (UCX by default; RDMA when the NICs support it)

References (checked Oct 2026):
  https://docs.vllm.ai/api/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.html
  https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
  https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html

Single-GPU variant: run with --role local to use LMCache only (no NIXL peer needed).
Disaggregated variant: start one process with --role prefill and one with --role decode, each on
its own GPU, with a proxy routing requests (see the vLLM NIXL guide). Each instance needs
VLLM_NIXL_SIDE_CHANNEL_HOST / VLLM_NIXL_SIDE_CHANNEL_PORT (default localhost:5600) set so the
peers can handshake.

Note: this sets LMCache through environment variables (LMCache's in-process mode), which LMCache
is deprecating in favor of its MP mode. run_vllm_lmcache.sh shows the current `vllm serve` path.
"""

import argparse
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# NixlConnector deprecates kv_role="kv_both"; give each instance an explicit role.
NIXL_ROLE = {"prefill": "kv_producer", "decode": "kv_consumer"}


def configure_lmcache(cpu_gib: float = 64, chunk_size: int = 256) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cpu_gib)  # GiB of pinned CPU DRAM


def build_kv_transfer_config(role: str) -> KVTransferConfig:
    lmcache = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    if role == "local":
        return KVTransferConfig(**lmcache)

    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                lmcache,  # listed first: preferred load source
                {"kv_connector": "NixlConnector", "kv_role": NIXL_ROLE[role]},
            ]
        },
    )


def create_tiered_kv_engine(model: str, role: str, cpu_gib: float) -> LLM:
    configure_lmcache(cpu_gib=cpu_gib)
    return LLM(
        model=model,
        kv_transfer_config=build_kv_transfer_config(role),
        max_model_len=16384,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
    )


def ttft_experiment(llm: LLM) -> None:
    """Same long context, different questions: the second and third runs should reuse KV."""
    context = "You are an AgentOS kernel assistant. " + ("Scheduler log line. " * 2000)
    params = SamplingParams(temperature=0.0, max_tokens=1)  # 1 token ~= time to first token
    for q in ["Summarize the log.", "Count the lines.", "Any errors?"]:
        start = time.perf_counter()
        llm.generate([context + "\nQuestion: " + q], params)
        print(f"{q:<22} {time.perf_counter() - start:6.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--role", choices=["local", "prefill", "decode"], default="local")
    ap.add_argument("--cpu-gib", type=float, default=64)
    ap.add_argument("--print-only", action="store_true", help="print the config without loading a model")
    args = ap.parse_args()

    ktc = build_kv_transfer_config(args.role)
    print(f"kv_connector={ktc.kv_connector} kv_role={ktc.kv_role} extra={ktc.kv_connector_extra_config}")
    if not args.print_only:
        ttft_experiment(create_tiered_kv_engine(args.model, args.role, args.cpu_gib))
