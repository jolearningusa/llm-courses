"""
Lab 12A (folder lab1_memory_plugins): your nanoos memory and context engine as a Hermes Agent plugin.

The adapters live in the lab kit (labs/kit/nanoos/ports.py, finished in Lab 12, Part A); this file is only the
plugin that loads them into Hermes with your Lab 3 engine and your Lab 4 MemoryStore.

API reference (checked 2026-10-27):
  - Memory providers: https://hermes-agent.nousresearch.com/docs/developer-guide/memory-provider-plugin
  - Context engines:  https://hermes-agent.nousresearch.com/docs/developer-guide/context-engine-plugin

Install as a user plugin (one plugin per directory):

    ~/.hermes/plugins/nanoos/
    ├── __init__.py   <- copy this file here
    └── plugin.yaml   <- see PLUGIN_YAML below

Point it at the kit and choose the engine with environment variables:

    export NANOOS_KIT=/path/to/llm-agentos-course/labs/kit
    export NANOOS_ENGINE=threshold        # or clearing
    export NANOOS_MODE=compress           # compress: Hermes rewrites the transcript; select: a view per request
    export NANOOS_MEMORY_MODEL=...        # optional: the model the extractor calls (default: the kit's API model)

Then activate in ~/.hermes/config.yaml (plugins are never auto-activated):

    memory:
      provider: nanoos-memory
    context:
      engine: nanoos-context

Check it loaded with `hermes memory status`.

`python hermes_custom_memory.py` runs a two-session demo offline through the kit's HermesHost (it needs your
Lab 3 and Lab 4 code, since it uses the real engines).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

KIT = Path(os.environ.get("NANOOS_KIT", Path(__file__).resolve().parent.parent / "kit"))
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from nanoos.memory import ModelExtractor, ScriptedExtractor  # noqa: E402
from nanoos.ports import HermesContextEngine, HermesMemoryProvider, make_engine  # noqa: E402

PLUGIN_YAML = """\
name: nanoos
version: 0.12.0
description: LLM AgentOS course - your nanoos memory store and context engine (Lab 12)
hooks:
  - prefetch
  - queue_prefetch
  - sync_turn
  - on_pre_compress
  - on_session_end
"""


def model_extractor():
    """Your Lab 4 extractor, calling the model in NANOOS_MEMORY_MODEL (or the kit's default API model)."""
    from models.openai_compat import OpenAICompatModel
    name = os.environ.get("NANOOS_MEMORY_MODEL")
    return ModelExtractor(OpenAICompatModel(model=name) if name else OpenAICompatModel())


def build(extractor_factory=model_extractor) -> tuple[HermesMemoryProvider, HermesContextEngine]:
    engine = HermesContextEngine(make_engine(os.environ.get("NANOOS_ENGINE", "threshold")),
                                 mode=os.environ.get("NANOOS_MODE", "compress"))
    return HermesMemoryProvider(extractor_factory), engine


def register(ctx) -> None:
    # Fresh instances per load. Hermes accepts one external memory provider and one context engine, and clones
    # the engine per agent with clone_for_agent().
    memory, engine = build()
    ctx.register_memory_provider(memory)
    if hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)


if __name__ == "__main__":
    import tempfile

    from models.scripted import ScriptedModel
    from nanoos.hosts import HermesHost
    from tasks.basic import say

    class NoTools:
        def schemas(self):
            return []

        def call(self, name, arguments):
            return f"ERROR: no tool {name}", True

    fact = "I moved to Lisbon last month."
    ops = {fact: [{"op": "ADD", "text": "Lives in Lisbon", "key": "home_city"}]}
    extract = lambda: ScriptedExtractor(ops)  # noqa: E731
    home = tempfile.mkdtemp(prefix="hermes-home-")
    for session, turn in (("s1", fact), ("s2", "Which city do I live in? Lisbon or Porto?")):
        memory, _ = build(extract)
        result = HermesHost(ScriptedModel([say("Noted.")]), NoTools(), memory=memory, hermes_home=home,
                            user_id="alice", session_id=session).run([turn])
        print(f"{session}: the model saw ->", result.requests[0][1]["content"].replace("\n", " | "))
