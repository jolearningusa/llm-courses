"""
Lab 1C: Letta multi-agent shared memory blocks.

In Letta (formerly MemGPT) a *memory block* is a labeled, size-limited piece of text that is
always in the agent's context window: the "core memory" or L1 of the MemGPT memory hierarchy.
Attaching the same block to several agents gives them shared memory: when one agent edits it,
the others see the new value on their next turn.

API reference (checked Oct 2026): https://docs.letta.com/guides/core-concepts/memory/shared-memory
Note: this uses the Python V1 SDK (`letta-client`), which Letta is deprecating in favor of the
TypeScript-only Letta Agent SDK. The calls below are the documented V1 calls.

Setup:
    pip install letta-client
    # Either Letta Cloud:
    export LETTA_API_KEY=...
    # or a self-hosted server:
    docker run -p 8283:8283 -e OPENAI_API_KEY=... letta/letta:latest
    export LETTA_BASE_URL=http://localhost:8283

Run:
    python letta_shared_blocks.py
"""

import os

from letta_client import Letta

MODEL = os.getenv("LETTA_MODEL", "openai/gpt-4.1")


def make_client() -> Letta:
    if os.getenv("LETTA_BASE_URL"):
        return Letta(base_url=os.environ["LETTA_BASE_URL"])
    return Letta(api_key=os.environ["LETTA_API_KEY"])


def setup_multi_agent_shared_memory(client: Letta):
    # 1. Create the shared block once. `limit` caps its size in characters, like a fixed-size page.
    shared_block = client.blocks.create(
        label="sprint_state",
        description="Shared coordination state for the supervisor and worker agents.",
        value="Phase: kernel scheduler benchmark | Blockers: none",
        limit=2000,
    )

    # 2. Attach the same block to both agents with block_ids.
    #    (Putting the block id inside memory_blocks would create a *new* block whose text is the id.)
    supervisor = client.agents.create(
        name="aios_supervisor",
        model=MODEL,
        memory_blocks=[{"label": "persona", "value": "I am the supervisor. I plan benchmarks and "
                                                     "record decisions in sprint_state."}],
        block_ids=[shared_block.id],
    )
    worker = client.agents.create(
        name="aios_worker",
        model=MODEL,
        memory_blocks=[{"label": "persona", "value": "I am the worker. I run vLLM KV-cache benchmarks "
                                                     "and read sprint_state before starting."}],
        block_ids=[shared_block.id],
    )
    print(f"Shared block {shared_block.id} attached to {supervisor.id} and {worker.id}")
    return shared_block, supervisor, worker


def demo_visibility(client: Letta, shared_block, supervisor, worker) -> None:
    # The supervisor edits the shared block through its built-in memory tools...
    client.agents.messages.create(
        agent_id=supervisor.id,
        messages=[{"role": "user", "content": "Update sprint_state: the blocker is "
                                              "'NIXL side channel port 5600 is closed'."}],
    )
    # ...and the worker sees the new value in its own context on its next turn.
    print("Block now reads:", client.blocks.retrieve(shared_block.id).value)
    reply = client.agents.messages.create(
        agent_id=worker.id,
        messages=[{"role": "user", "content": "What is the current blocker?"}],
    )
    for m in reply.messages:
        if getattr(m, "message_type", "") == "assistant_message":
            print("Worker:", m.content)


if __name__ == "__main__":
    c = make_client()
    blk, sup, wrk = setup_multi_agent_shared_memory(c)
    demo_visibility(c, blk, sup, wrk)
